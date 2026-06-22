// Copyright 2024 KVCache.AI
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

#include "tent/runtime/segment_registry.h"

#include <cassert>
#include <set>

#include "tent/common/status.h"
#include "tent/common/utils/ip.h"
#include "tent/common/utils/os.h"
#include "tent/runtime/control_plane.h"

namespace mooncake {
namespace tent {

namespace {

bool isDirectConnectEndpoint(const std::string& name) {
    if (name.empty()) {
        return false;
    }
    const auto colon = name.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= name.size()) {
        return false;
    }
    auto [host, port] = parseHostNameWithPort(name, 0);
    return !host.empty() && port != 0;
}

std::pair<std::string, std::string> parseStorageConnString(
    const std::string& storage_conn_string) {
    std::pair<std::string, std::string> result{"etcd", storage_conn_string};
    const std::size_t pos = storage_conn_string.find("://");
    if (pos != std::string::npos) {
        result.first = storage_conn_string.substr(0, pos);
        if (result.first == "http" || result.first == "https") {
            result.second = storage_conn_string;
        } else {
            result.second = storage_conn_string.substr(pos + 3);
        }
    }
    return result;
}

}  // namespace

static inline std::string getFullMetadataKey(const std::string &segment_name) {
    const static std::string kCommonKeyPrefix = "mooncake/tent/";
    return kCommonKeyPrefix + segment_name;
}

CentralSegmentRegistry::CentralSegmentRegistry(const std::string &type,
                                               const std::string &servers) {
    plugin_ = MetaStore::Create(type, servers);
}

Status CentralSegmentRegistry::getSegmentDesc(SegmentDescRef &desc,
                                              const std::string &segment_name) {
    if (!plugin_)
        return Status::MetadataError(
            "Central metadata store not started" LOC_MARK);
    std::string jstr;
    desc = nullptr;

    auto status = plugin_->get(getFullMetadataKey(segment_name), jstr);
    if (!status.ok()) return status;
    desc = std::make_shared<SegmentDesc>();
    *desc = json::parse(jstr).get<SegmentDesc>();
    return Status::OK();
}

Status CentralSegmentRegistry::putSegmentDesc(SegmentDescRef &desc) {
    if (!plugin_)
        return Status::MetadataError(
            "Central metadata store not started" LOC_MARK);
    json j = *desc;
    return plugin_->set(getFullMetadataKey(desc->name), j.dump());
}

Status CentralSegmentRegistry::deleteSegmentDesc(
    const std::string &segment_name) {
    if (!plugin_)
        return Status::MetadataError(
            "Central metadata store not started" LOC_MARK);
    return plugin_->remove(getFullMetadataKey(segment_name));
}

Status PeerSegmentRegistry::getSegmentDesc(SegmentDescRef &desc,
                                           const std::string &segment_name) {
    std::string response;
    if (segment_name.empty())
        return Status::InvalidArgument("Empty segment name" LOC_MARK);
    CHECK_STATUS(ControlClient::getSegmentDesc(segment_name, response));
    if (response.empty()) {
        return Status::InvalidEntry(std::string("Segment ") + segment_name +
                                    "not found" + LOC_MARK);
    }
    desc = std::make_shared<SegmentDesc>();
    *desc = json::parse(response).get<SegmentDesc>();
    return Status::OK();
}

HybridSegmentRegistry::HybridSegmentRegistry(
    const std::string &storage_conn_string) {
    auto [type, servers] = parseStorageConnString(storage_conn_string);
    central_ = std::make_unique<CentralSegmentRegistry>(type, servers);
}

Status HybridSegmentRegistry::getSegmentDesc(SegmentDescRef &desc,
                                             const std::string &segment_name) {
    auto status = central_->getSegmentDesc(desc, segment_name);
    if (status.ok()) {
        return status;
    }
    if (!isDirectConnectEndpoint(segment_name)) {
        return status;
    }
    LOG(INFO) << "Hybrid metadata: falling back to P2P segment lookup for "
              << segment_name;
    PeerSegmentRegistry peer;
    return peer.getSegmentDesc(desc, segment_name);
}

Status HybridSegmentRegistry::putSegmentDesc(SegmentDescRef &desc) {
    if (desc) {
        if (desc->discovery.mode.empty()) {
            desc->discovery.mode = "hybrid";
        }
        if (desc->discovery.rpc_endpoint.empty() &&
            !desc->rpc_server_addr.empty()) {
            desc->discovery.rpc_endpoint = desc->rpc_server_addr;
        }
        if (desc->discovery.prefer.empty()) {
            desc->discovery.prefer = "central";
        }
    }
    return central_->putSegmentDesc(desc);
}

Status HybridSegmentRegistry::deleteSegmentDesc(
    const std::string &segment_name) {
    return central_->deleteSegmentDesc(segment_name);
}

}  // namespace tent
}  // namespace mooncake
