/**
 * Simple JSON Parser Implementation
 * Contains static member definitions to avoid multiple definition errors
 */

#include "simple_json.h"

namespace Json {
    // Define the static null value to avoid multiple definition errors
    Value Value::null;
}
