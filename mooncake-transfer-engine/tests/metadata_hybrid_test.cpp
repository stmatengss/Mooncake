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
#include "transfer_metadata.h"
#include "transfer_metadata_plugin.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

using namespace mooncake;

namespace {

class GlogEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        google::InitGoogleLogging("MetadataHybridTest");
        FLAGS_logtostderr = 1;
    }
    void TearDown() override { google::ShutdownGoogleLogging(); }
};

uint16_t pickAvailablePort(int& sockfd) {
    sockfd = -1;
    return findAvailableTcpPort(sockfd);
}

std::string buildEndpoint(const std::string& ip, uint16_t port) {
    return ip + ":" + std::to_string(port);
}

void setupTcpSegment(TransferMetadata& metadata, const std::string& segment_name) {
    auto desc = std::make_shared<TransferMetadata::SegmentDesc>();
    desc->name = segment_name;
    desc->protocol = "tcp";
    desc->tcp_data_port = 0;

    TransferMetadata::BufferDesc buffer;
    buffer.name = "cpu:0";
    buffer.addr = 0x1000;
    buffer.length = 4096;
    desc->buffers.push_back(buffer);

    ASSERT_EQ(metadata.addLocalSegment(LOCAL_SEGMENT_ID, segment_name,
                                     std::move(desc)),
              0);
}

std::string startMetadataRpcDaemon(TransferMetadata& metadata,
                                   const std::string& bind_ip = "127.0.0.1") {
    TransferMetadata::RpcMetaDesc rpc_desc;
    rpc_desc.ip_or_host_name = bind_ip;
    rpc_desc.rpc_port = pickAvailablePort(rpc_desc.sockfd);
    if (rpc_desc.rpc_port == 0) {
        ADD_FAILURE() << "Failed to pick RPC port";
        return {};
    }
    const std::string server_name =
        buildEndpoint(rpc_desc.ip_or_host_name, rpc_desc.rpc_port);
    if (metadata.addRpcMetaEntry(server_name, rpc_desc) != 0) {
        ADD_FAILURE() << "Failed to start metadata RPC daemon on " << server_name;
        return {};
    }
    return server_name;
}

bool waitForRpcReady(TransferMetadata& metadata,
                     const std::string& server_name, int attempts = 100) {
    for (int i = 0; i < attempts; ++i) {
        if (metadata.sendProbe(server_name) == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return false;
}

#ifndef MOONCAKE_TEST_HTTP_SERVER_SCRIPT
#define MOONCAKE_TEST_HTTP_SERVER_SCRIPT \
    "mooncake-transfer-engine/tests/http_metadata_test_server.py"
#endif

bool isTcpPortOpen(const std::string& ip, uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
    const bool open = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    close(fd);
    return open;
}

void stopHttpMetadataServer(uint16_t port) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "pkill -f 'http_metadata_test_server.py %u' 2>/dev/null",
             static_cast<unsigned>(port));
    std::system(cmd);
}

class HttpMetadataServerFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        int sockfd = -1;
        port_ = pickAvailablePort(sockfd);
        if (sockfd >= 0) {
            close(sockfd);
        }
        ASSERT_GT(port_, 0);

        char port_buf[16];
        snprintf(port_buf, sizeof(port_buf), "%u", port_);
        char start_cmd[512];
        snprintf(start_cmd, sizeof(start_cmd),
                 "python3 %s %s >/dev/null 2>&1 &",
                 MOONCAKE_TEST_HTTP_SERVER_SCRIPT, port_buf);
        ASSERT_EQ(std::system(start_cmd), 0);

        bool ready = false;
        for (int i = 0; i < 100; ++i) {
            if (isTcpPortOpen("127.0.0.1", port_)) {
                ready = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ASSERT_TRUE(ready) << "HTTP metadata test server failed to start";

        base_url_ = "http://127.0.0.1:" + std::to_string(port_) + "/metadata";
    }

    void TearDown() override { stopHttpMetadataServer(port_); }

    uint16_t port_{0};
    std::string base_url_;
};

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new GlogEnvironment());
    return RUN_ALL_TESTS();
}

TEST(MetadataHybridTest, P2pExchangeBetweenPeers) {
    TransferMetadata peer_a(P2PHANDSHAKE);
    TransferMetadata peer_b(P2PHANDSHAKE);

    setupTcpSegment(peer_a, "peer_a");
    setupTcpSegment(peer_b, "peer_b");

    const std::string endpoint_a = startMetadataRpcDaemon(peer_a);
    const std::string endpoint_b = startMetadataRpcDaemon(peer_b);
    ASSERT_FALSE(endpoint_a.empty());
    ASSERT_FALSE(endpoint_b.empty());

    ASSERT_TRUE(waitForRpcReady(peer_a, endpoint_b));
    ASSERT_TRUE(waitForRpcReady(peer_b, endpoint_a));

    auto desc_from_b = peer_a.getSegmentDesc(endpoint_b);
    ASSERT_NE(desc_from_b, nullptr);
    EXPECT_EQ(desc_from_b->name, "peer_b");
    EXPECT_EQ(desc_from_b->protocol, "tcp");
    EXPECT_FALSE(desc_from_b->buffers.empty());

    auto desc_from_a = peer_b.getSegmentDesc(endpoint_a);
    ASSERT_NE(desc_from_a, nullptr);
    EXPECT_EQ(desc_from_a->name, "peer_a");
}

TEST(MetadataHybridTest, HybridFallbackToDirectP2pEndpoint) {
    TransferMetadata p2p_target(P2PHANDSHAKE);
    setupTcpSegment(p2p_target, "p2p_target");
    const std::string target_endpoint = startMetadataRpcDaemon(p2p_target);
    ASSERT_FALSE(target_endpoint.empty());
    ASSERT_TRUE(waitForRpcReady(p2p_target, target_endpoint));

    int dead_sockfd = -1;
    const uint16_t dead_http_port = pickAvailablePort(dead_sockfd);
    const std::string hybrid_conn =
        "http://127.0.0.1:" + std::to_string(dead_http_port) +
        "/metadata+P2PHANDSHAKE";

    TransferMetadata hybrid_initiator(hybrid_conn);
    setupTcpSegment(hybrid_initiator, "initiator");

    auto desc = hybrid_initiator.getSegmentDesc(target_endpoint);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->name, "p2p_target");
}

TEST_F(HttpMetadataServerFixture, HybridPublishAndCentralLookup) {
    const std::string hybrid_conn = base_url_ + "+P2PHANDSHAKE";
    const std::string central_conn = base_url_;

    TransferMetadata hybrid_node(hybrid_conn);
    const std::string segment_name = "hybrid_node_test";

    setupTcpSegment(hybrid_node, segment_name);
    const std::string rpc_endpoint = startMetadataRpcDaemon(hybrid_node);
    ASSERT_FALSE(rpc_endpoint.empty());

    ASSERT_EQ(hybrid_node.updateLocalSegmentDesc(), 0);

    TransferMetadata central_reader(central_conn);
    auto desc = central_reader.getSegmentDescByName(segment_name, true);
    ASSERT_NE(desc, nullptr) << "Central lookup of hybrid-published segment failed";
    EXPECT_EQ(desc->name, segment_name);
    EXPECT_EQ(desc->protocol, "tcp");
    EXPECT_EQ(desc->discovery.mode, "hybrid");
    EXPECT_EQ(desc->discovery.rpc_endpoint, rpc_endpoint);

    TransferMetadata::RpcMetaDesc rpc_meta;
    ASSERT_EQ(central_reader.getRpcMetaEntry(rpc_endpoint, rpc_meta), 0);
    const auto& local_rpc = hybrid_node.localRpcMeta();
    EXPECT_EQ(rpc_meta.ip_or_host_name, local_rpc.ip_or_host_name);
    EXPECT_EQ(rpc_meta.rpc_port, local_rpc.rpc_port);
}

TEST_F(HttpMetadataServerFixture, CentralInitiatorReadsHybridTargetByHostname) {
    const std::string hybrid_conn = base_url_ + "+P2PHANDSHAKE";
    const std::string central_conn = base_url_;
    const std::string segment_name = "node_central_to_hybrid";

    TransferMetadata hybrid_target(hybrid_conn);
    setupTcpSegment(hybrid_target, segment_name);
    startMetadataRpcDaemon(hybrid_target);
    ASSERT_EQ(hybrid_target.updateLocalSegmentDesc(), 0);

    TransferMetadata central_initiator(central_conn);
    auto desc = central_initiator.getSegmentDescByName(segment_name, true);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->name, segment_name);
    EXPECT_FALSE(desc->discovery.rpc_endpoint.empty());

    TransferMetadata::RpcMetaDesc rpc_meta;
    ASSERT_EQ(
        central_initiator.getRpcMetaEntry(desc->discovery.rpc_endpoint, rpc_meta),
        0);
    EXPECT_GT(rpc_meta.rpc_port, 0);
}

TEST_F(HttpMetadataServerFixture, HybridInitiatorUsesP2pFallbackForDirectEndpoint) {
    const std::string hybrid_conn = base_url_ + "+P2PHANDSHAKE";

    TransferMetadata p2p_only(P2PHANDSHAKE);
    setupTcpSegment(p2p_only, "p2p_only_target");
    const std::string p2p_endpoint = startMetadataRpcDaemon(p2p_only);
    ASSERT_FALSE(p2p_endpoint.empty());
    ASSERT_TRUE(waitForRpcReady(p2p_only, p2p_endpoint));

    TransferMetadata hybrid_initiator(hybrid_conn);
    setupTcpSegment(hybrid_initiator, "hybrid_initiator");

    auto desc = hybrid_initiator.getSegmentDescByName(p2p_endpoint, true);
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->name, "p2p_only_target");
}

TEST(MetadataHybridTest, PureP2pDoesNotPublishToCentral) {
    TransferMetadata p2p(P2PHANDSHAKE);
    setupTcpSegment(p2p, "pure_p2p_node");
    startMetadataRpcDaemon(p2p);
    EXPECT_EQ(p2p.updateLocalSegmentDesc(), 0);
}
