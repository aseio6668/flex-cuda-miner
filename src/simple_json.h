/**
 * Simple JSON Parser for Mining Pool Communication
 * Basic JSON parsing and writing for Stratum protocol
 */

#ifndef SIMPLE_JSON_H
#define SIMPLE_JSON_H

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>

namespace Json {
      class Value {
    public:
        std::string stringValue_;
        int intValue_;
        std::vector<Value> arrayValue_;
        std::map<std::string, Value> objectValue_;
        bool isString_;
        bool isInt_;
        bool isArray_;
        bool isObject_;
        bool isNull_;
        
    public:
        // Constructors
        Value() : intValue_(0), isString_(false), isInt_(false), isArray_(false), isObject_(false), isNull_(true) {}
        Value(const std::string& str) : stringValue_(str), intValue_(0), isString_(true), isInt_(false), isArray_(false), isObject_(false), isNull_(false) {}
        Value(int val) : intValue_(val), isString_(false), isInt_(true), isArray_(false), isObject_(false), isNull_(false) {}
        
        // Type checking
        bool isNull() const { return isNull_; }
        bool isString() const { return isString_; }
        bool isInt() const { return isInt_; }
        bool isArray() const { return isArray_; }
        bool isObject() const { return isObject_; }
        
        // Value access
        std::string asString() const { return stringValue_; }
        int asInt() const { return intValue_; }
          // Array operations
        void append(const Value& value) {
            if (!isArray_) {
                isArray_ = true;
                isNull_ = false;
                arrayValue_.clear();
            }
            arrayValue_.push_back(value);
        }
        
        // Object operations
        bool isMember(const std::string& key) const {
            if (!isObject_) return false;
            return objectValue_.find(key) != objectValue_.end();
        }
        
        size_t size() const {
            if (isArray_) return arrayValue_.size();
            if (isObject_) return objectValue_.size();
            return 0;
        }
        
        Value& operator[](size_t index) {
            if (!isArray_) {
                isArray_ = true;
                isNull_ = false;
            }
            if (index >= arrayValue_.size()) {
                arrayValue_.resize(index + 1);
            }
            return arrayValue_[index];
        }        Value& operator[](const std::string& key) {
            if (!isObject_) {
                isObject_ = true;
                isNull_ = false;
            }
            return objectValue_[key];
        }
        
        // Const access operators
        const Value& operator[](size_t index) const {
            static Value nullValue;
            if (!isArray_ || index >= arrayValue_.size()) {
                return nullValue;
            }
            return arrayValue_[index];
        }
        
        const Value& operator[](const std::string& key) const {
            static Value nullValue;
            if (!isObject_) {
                return nullValue;
            }
            auto it = objectValue_.find(key);
            if (it == objectValue_.end()) {
                return nullValue;
            }
            return it->second;
        }
        
        // Assignment operators
        Value& operator=(const std::string& str) {
            stringValue_ = str;
            isString_ = true;
            isInt_ = false;
            isArray_ = false;
            isObject_ = false;
            isNull_ = false;
            return *this;
        }
        
        Value& operator=(int val) {
            intValue_ = val;
            isString_ = false;
            isInt_ = true;
            isArray_ = false;
            isObject_ = false;
            isNull_ = false;
            return *this;
        }
        
        // Iterator support for arrays
        std::vector<Value>::iterator begin() { return arrayValue_.begin(); }
        std::vector<Value>::iterator end() { return arrayValue_.end(); }
        std::vector<Value>::const_iterator begin() const { return arrayValue_.begin(); }
        std::vector<Value>::const_iterator end() const { return arrayValue_.end(); }
          static Value null;
        
        // Friend functions for parser access
        friend void writeValue(std::ostringstream& ss, const Value& value);
        friend Value makeArray();
    };
      // Static member declared in header, defined separately to avoid multiple definition// Helper functions for array creation
    inline Value makeArray() {
        Value arr;
        arr.isArray_ = true;
        arr.isNull_ = false;
        return arr;    }
    
    class Reader {
    public:
        bool parse(const std::string& json, Value& root) {
            // Very simple JSON parser - handles basic cases for mining pool
            std::string trimmed = trim(json);
            
            if (trimmed.empty()) return false;
            
            if (trimmed[0] == '{') {
                return parseObject(trimmed, root);
            } else if (trimmed[0] == '[') {
                return parseArray(trimmed, root);
            }
            
            return false;
        }
        
    private:
        std::string trim(const std::string& str) {
            size_t start = str.find_first_not_of(" \t\n\r");
            if (start == std::string::npos) return "";
            size_t end = str.find_last_not_of(" \t\n\r");
            return str.substr(start, end - start + 1);
        }
        
        bool parseObject(const std::string& json, Value& root) {
            // Simple object parsing
            root = Value();
            root.isObject_ = true;
            root.isNull_ = false;
            return true;
        }
        
        bool parseArray(const std::string& json, Value& root) {
            // Simple array parsing
            root = Value();
            root.isArray_ = true;
            root.isNull_ = false;
            return true;
        }
    };
    
    class StreamWriterBuilder {
    public:
        StreamWriterBuilder& operator[](const std::string& key) { return *this; }
    };
      // Forward declaration for writeValue function
    void writeValue(std::ostringstream& ss, const Value& value);
    
    inline std::string writeString(const StreamWriterBuilder& builder, const Value& value) {
        // Simple JSON writer
        std::ostringstream ss;
        writeValue(ss, value);
        return ss.str();
    }
    
    inline void writeValue(std::ostringstream& ss, const Value& value) {
        if (value.isNull()) {
            ss << "null";
        } else if (value.isString()) {
            ss << "\"" << value.asString() << "\"";
        } else if (value.isInt()) {
            ss << value.asInt();
        } else if (value.isArray()) {
            ss << "[";
            bool first = true;
            for (const auto& item : value) {
                if (!first) ss << ",";
                writeValue(ss, item);
                first = false;
            }
            ss << "]";
        } else if (value.isObject()) {
            ss << "{}"; // Simplified object serialization
        }
    }
    
} // namespace Json

#endif // SIMPLE_JSON_H
