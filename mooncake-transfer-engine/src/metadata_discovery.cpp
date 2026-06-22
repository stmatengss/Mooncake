// Copyright 2026 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "metadata_discovery.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

#include "common.h"

namespace mooncake {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

MetadataDiscoveryMode parseDiscoveryModeEnv() {
    const char* env = std::getenv("MC_METADATA_DISCOVERY_MODE");
    if (env == nullptr || std::strlen(env) == 0) {
        return MetadataDiscoveryMode::Central;
    }
    return metadataDiscoveryModeFromString(env);
}

bool endsWithSuffix(const std::string& value, const char* suffix) {
    const size_t suffix_len = std::strlen(suffix);
    if (value.size() < suffix_len) return false;
    return value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

}  // namespace

const char* metadataDiscoveryModeToString(MetadataDiscoveryMode mode) {
    switch (mode) {
        case MetadataDiscoveryMode::Central:
            return "central";
        case MetadataDiscoveryMode::P2P:
            return "p2p";
        case MetadataDiscoveryMode::Hybrid:
            return "hybrid";
    }
    return "central";
}

MetadataDiscoveryMode metadataDiscoveryModeFromString(const std::string& value) {
    const std::string normalized = toLower(value);
    if (normalized == "p2p" || normalized == "p2phandshake") {
        return MetadataDiscoveryMode::P2P;
    }
    if (normalized == "hybrid") {
        return MetadataDiscoveryMode::Hybrid;
    }
    return MetadataDiscoveryMode::Central;
}

ParsedMetadataConnString parseMetadataConnString(
    const std::string& conn_string) {
    ParsedMetadataConnString parsed;
    parsed.storage_conn_string = conn_string;

    MetadataDiscoveryMode env_mode = parseDiscoveryModeEnv();
    const bool env_explicit =
        std::getenv("MC_METADATA_DISCOVERY_MODE") != nullptr &&
        std::strlen(std::getenv("MC_METADATA_DISCOVERY_MODE")) > 0;

    if (conn_string == P2PHANDSHAKE) {
        parsed.storage_conn_string = P2PHANDSHAKE;
        parsed.discovery_mode = MetadataDiscoveryMode::P2P;
        return parsed;
    }

    if (endsWithSuffix(conn_string, P2PHANDSHAKE_SUFFIX)) {
        parsed.storage_conn_string =
            conn_string.substr(0, conn_string.size() - std::strlen(P2PHANDSHAKE_SUFFIX));
        parsed.discovery_mode = MetadataDiscoveryMode::Hybrid;
        return parsed;
    }

    if (env_explicit) {
        parsed.discovery_mode = env_mode;
        return parsed;
    }

    parsed.discovery_mode = MetadataDiscoveryMode::Central;
    return parsed;
}

}  // namespace mooncake
