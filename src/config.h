/**
 * Configuration file parser for Lyncoin Flex CUDA Miner
 * Supports INI-style configuration files
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

class Config {
private:
    std::map<std::string, std::string> values;
    
    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(' ');
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }
    
public:
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open config file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            // Parse key=value pairs
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                values[key] = value;
            }
        }
        
        file.close();
        return true;
    }
    
    std::string getString(const std::string& key, const std::string& defaultValue = "") {
        auto it = values.find(key);
        return (it != values.end()) ? it->second : defaultValue;
    }
    
    int getInt(const std::string& key, int defaultValue = 0) {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                std::cerr << "Warning: Invalid integer value for " << key << ": " << it->second << std::endl;
            }
        }
        return defaultValue;
    }
    
    double getDouble(const std::string& key, double defaultValue = 0.0) {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stod(it->second);
            } catch (...) {
                std::cerr << "Warning: Invalid double value for " << key << ": " << it->second << std::endl;
            }
        }
        return defaultValue;
    }
    
    bool getBool(const std::string& key, bool defaultValue = false) {
        auto it = values.find(key);
        if (it != values.end()) {
            std::string value = it->second;
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return (value == "true" || value == "1" || value == "yes" || value == "on");
        }
        return defaultValue;
    }
    
    void setValue(const std::string& key, const std::string& value) {
        values[key] = value;
    }
    
    void printAll() {
        std::cout << "Configuration values:" << std::endl;
        for (const auto& pair : values) {
            std::cout << "  " << pair.first << " = " << pair.second << std::endl;
        }
    }
};

// Global configuration instance
extern Config g_config;

#endif // CONFIG_H
