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

#ifndef METADATA_DISCOVERY_H
#define METADATA_DISCOVERY_H

#include <string>

#include "common.h"

namespace mooncake {

#define P2PHANDSHAKE "P2PHANDSHAKE"
#define P2PHANDSHAKE_SUFFIX "+P2PHANDSHAKE"

enum class MetadataDiscoveryMode {
    Central,
    P2P,
    Hybrid,
};

struct ParsedMetadataConnString {
    std::string storage_conn_string;
    MetadataDiscoveryMode discovery_mode{MetadataDiscoveryMode::Central};
};

// Parse metadata connection string and resolve discovery mode.
// Supports: P2PHANDSHAKE, etcd://host:2379+P2PHANDSHAKE, MC_METADATA_DISCOVERY_MODE.
ParsedMetadataConnString parseMetadataConnString(
    const std::string& conn_string);

const char* metadataDiscoveryModeToString(MetadataDiscoveryMode mode);

MetadataDiscoveryMode metadataDiscoveryModeFromString(const std::string& value);

// True when name is host:port (or [ipv6]:port) with an explicit port suitable
// for direct P2P metadata exchange.
inline bool isDirectConnectEndpoint(const std::string& name) {
    if (name.empty() || !hasExplicitPort(name)) {
        return false;
    }
    auto [host, port] = parseHostNameWithPort(name);
    return !host.empty() && port != 0;
}

}  // namespace mooncake

#endif  // METADATA_DISCOVERY_H
