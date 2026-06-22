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

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "tent/common/config.h"
#include "tent/runtime/segment_registry.h"
#include "tent/runtime/transfer_engine_impl.h"

#ifndef _WIN32
#include <asio.hpp>
#include <ylt/coro_http/coro_http_server.hpp>
#endif

namespace mooncake {
namespace tent {
namespace {

constexpr char kLoopbackHostname[] = "127.0.0.1";
constexpr char kMetadataKeyPrefix[] = "mooncake/tent/";

#ifndef _WIN32

uint16_t reserveUnusedTcpPort() {
    asio::io_context io_context;
    asio::ip::tcp::acceptor acceptor(
        io_context,
        asio::ip::tcp::endpoint(asio::ip::address_v4::loopback(), 0));
    return acceptor.local_endpoint().port();
}

std::string buildHttpMetadataEndpoint(uint16_t port) {
    return std::string("http://") + kLoopbackHostname + ":" +
           std::to_string(port) + "/metadata";
}

std::string buildMetadataKey(const std::string& segment_name) {
    return std::string(kMetadataKeyPrefix) + segment_name;
}

class TestHttpMetadataServer {
   public:
    explicit TestHttpMetadataServer(uint16_t port)
        : port_(port),
          server_(std::make_unique<coro_http::coro_http_server>(1, port)) {
        initServer();
    }

    ~TestHttpMetadataServer() { stop(); }

    bool start() {
        if (started_) {
            return running_;
        }
        server_->async_start();
        started_ = true;
        for (int attempt = 0; attempt < 50; ++attempt) {
            if (isReachable()) {
                running_ = true;
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        stop();
        return false;
    }

    void stop() {
        if (!started_) {
            return;
        }
        server_->stop();
        started_ = false;
        running_ = false;
    }

    std::optional<std::string> getStoredValue(const std::string& key) const {
        std::lock_guard<std::mutex> lock(store_mutex_);
        auto it = store_.find(key);
        if (it == store_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

   private:
    void initServer() {
        using namespace coro_http;

        server_->set_http_handler<GET>(
            "/metadata",
            [this](coro_http_request& req, coro_http_response& resp) {
                auto key = req.get_decode_query_value("key");
                if (key.empty()) {
                    resp.set_status_and_content(status_type::bad_request,
                                                "Missing key parameter");
                    return;
                }
                std::lock_guard<std::mutex> lock(store_mutex_);
                auto it = store_.find(key);
                if (it == store_.end()) {
                    resp.set_status_and_content(status_type::not_found,
                                                "metadata not found");
                    return;
                }
                resp.add_header("Content-Type", "application/json");
                resp.set_status_and_content(status_type::ok, it->second);
            });

        server_->set_http_handler<PUT>(
            "/metadata",
            [this](coro_http_request& req, coro_http_response& resp) {
                auto key = req.get_decode_query_value("key");
                if (key.empty()) {
                    resp.set_status_and_content(status_type::bad_request,
                                                "Missing key parameter");
                    return;
                }
                std::lock_guard<std::mutex> lock(store_mutex_);
                store_[key] = std::string(req.get_body());
                resp.set_status_and_content(status_type::ok,
                                            "metadata updated");
            });
    }

    bool isReachable() const {
        asio::io_context io_context;
        asio::ip::tcp::socket socket(io_context);
        std::error_code ec;
        socket.connect(
            asio::ip::tcp::endpoint(asio::ip::address_v4::loopback(), port_),
            ec);
        return !ec;
    }

    uint16_t port_;
    bool started_ = false;
    bool running_ = false;
    mutable std::mutex store_mutex_;
    std::unordered_map<std::string, std::string> store_;
    std::unique_ptr<coro_http::coro_http_server> server_;
};

SegmentDescRef makeMemorySegment(const std::string& name,
                                 const std::string& rpc_endpoint) {
    auto desc = std::make_shared<SegmentDesc>();
    desc->name = name;
    desc->type = SegmentType::Memory;
    desc->machine_id = "test-machine";
    desc->rpc_server_addr = rpc_endpoint;
    desc->detail = MemorySegmentDesc{};
    return desc;
}

void configureTcpOnlyTransports(Config& config) {
    config.set("transports/tcp/enable", true);
    config.set("transports/shm/enable", false);
    config.set("transports/rdma/enable", false);
    config.set("transports/io_uring/enable", false);
    config.set("transports/nvlink/enable", false);
    config.set("transports/mnnvl/enable", false);
    config.set("transports/gds/enable", false);
    config.set("transports/ascend_direct/enable", false);
}

#endif  // _WIN32

}  // namespace

TEST(TentHybridSegmentRegistryTest, PublishAndCentralLookup) {
#ifndef _WIN32
    const auto port = reserveUnusedTcpPort();
    TestHttpMetadataServer metadata_server(port);
    ASSERT_TRUE(metadata_server.start());

    const std::string endpoint = buildHttpMetadataEndpoint(port);
    const std::string segment_name = "tent_hybrid_node";
    const std::string rpc_endpoint =
        std::string(kLoopbackHostname) + ":15001";

    HybridSegmentRegistry registry(endpoint);
    auto desc = makeMemorySegment(segment_name, rpc_endpoint);
    ASSERT_TRUE(registry.putSegmentDesc(desc).ok());

    SegmentDescRef loaded;
    ASSERT_TRUE(registry.getSegmentDesc(loaded, segment_name).ok());
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->name, segment_name);
    EXPECT_EQ(loaded->discovery.mode, "hybrid");
    EXPECT_EQ(loaded->discovery.rpc_endpoint, rpc_endpoint);
    EXPECT_EQ(loaded->discovery.prefer, "central");

    auto stored = metadata_server.getStoredValue(buildMetadataKey(segment_name));
    ASSERT_TRUE(stored.has_value());
    auto stored_json = json::parse(*stored);
    EXPECT_EQ(stored_json.at("discovery").at("mode").get<std::string>(),
              "hybrid");
#else
    GTEST_SKIP() << "Requires local HTTP metadata server support";
#endif
}

TEST(TentHybridSegmentRegistryTest, CentralMissWithoutDirectEndpointFails) {
#ifndef _WIN32
    const auto port = reserveUnusedTcpPort();
    TestHttpMetadataServer metadata_server(port);
    ASSERT_TRUE(metadata_server.start());

    HybridSegmentRegistry registry(buildHttpMetadataEndpoint(port));
    SegmentDescRef loaded;
    auto status = registry.getSegmentDesc(loaded, "unknown_hostname");
    EXPECT_FALSE(status.ok());
#else
    GTEST_SKIP() << "Requires local HTTP metadata server support";
#endif
}

TEST(TentHybridSegmentRegistryTest, TransferEngineHybridInitPublishesToCentral) {
#ifndef _WIN32
    const auto port = reserveUnusedTcpPort();
    TestHttpMetadataServer metadata_server(port);
    ASSERT_TRUE(metadata_server.start());

    const std::string endpoint = buildHttpMetadataEndpoint(port);
    const std::string segment_name = "tent_hybrid_engine";
    const auto metadata_key = buildMetadataKey(segment_name);

    auto config = std::make_shared<Config>();
    config->set("metadata_type", "hybrid");
    config->set("metadata_servers", endpoint);
    config->set("local_segment_name", segment_name);
    config->set("rpc_server_hostname", kLoopbackHostname);
    config->set("rpc_server_port", "0");
    configureTcpOnlyTransports(*config);

    {
        TransferEngineImpl engine(config);
        ASSERT_TRUE(engine.available());
        EXPECT_EQ(engine.getSegmentName(), segment_name);

        auto stored = metadata_server.getStoredValue(metadata_key);
        ASSERT_TRUE(stored.has_value());
        auto stored_json = json::parse(*stored);
        EXPECT_EQ(stored_json.at("name").get<std::string>(), segment_name);
        EXPECT_EQ(stored_json.at("discovery").at("mode").get<std::string>(),
                  "hybrid");
        EXPECT_FALSE(
            stored_json.at("discovery").at("rpc_endpoint").get<std::string>()
                .empty());
    }
#else
    GTEST_SKIP() << "Requires local HTTP metadata server support";
#endif
}

}  // namespace tent
}  // namespace mooncake
