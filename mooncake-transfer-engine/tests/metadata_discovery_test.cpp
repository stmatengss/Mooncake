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

#include <gtest/gtest.h>

#include <cstdlib>

using namespace mooncake;

class MetadataDiscoveryTest : public ::testing::Test {
   protected:
    void TearDown() override { unsetenv("MC_METADATA_DISCOVERY_MODE"); }
};

TEST_F(MetadataDiscoveryTest, ParseP2PConnString) {
    auto parsed = parseMetadataConnString(P2PHANDSHAKE);
    EXPECT_EQ(parsed.storage_conn_string, P2PHANDSHAKE);
    EXPECT_EQ(parsed.discovery_mode, MetadataDiscoveryMode::P2P);
}

TEST_F(MetadataDiscoveryTest, ParseHybridConnStringSuffix) {
    auto parsed = parseMetadataConnString("etcd://10.0.0.1:2379+P2PHANDSHAKE");
    EXPECT_EQ(parsed.storage_conn_string, "etcd://10.0.0.1:2379");
    EXPECT_EQ(parsed.discovery_mode, MetadataDiscoveryMode::Hybrid);
}

TEST_F(MetadataDiscoveryTest, ParseCentralConnString) {
    auto parsed = parseMetadataConnString("etcd://10.0.0.1:2379");
    EXPECT_EQ(parsed.storage_conn_string, "etcd://10.0.0.1:2379");
    EXPECT_EQ(parsed.discovery_mode, MetadataDiscoveryMode::Central);
}

TEST_F(MetadataDiscoveryTest, ParseDiscoveryModeEnvOverridesCentral) {
    setenv("MC_METADATA_DISCOVERY_MODE", "hybrid", 1);
    auto parsed = parseMetadataConnString("etcd://10.0.0.1:2379");
    EXPECT_EQ(parsed.discovery_mode, MetadataDiscoveryMode::Hybrid);
}

TEST_F(MetadataDiscoveryTest, HybridSuffixTakesPrecedenceOverEnv) {
    setenv("MC_METADATA_DISCOVERY_MODE", "central", 1);
    auto parsed = parseMetadataConnString("http://10.0.0.1:8080/metadata+P2PHANDSHAKE");
    EXPECT_EQ(parsed.discovery_mode, MetadataDiscoveryMode::Hybrid);
}

TEST_F(MetadataDiscoveryTest, IsDirectConnectEndpoint) {
    EXPECT_TRUE(isDirectConnectEndpoint("127.0.0.1:12345"));
    EXPECT_TRUE(isDirectConnectEndpoint("[::1]:12345"));
    EXPECT_FALSE(isDirectConnectEndpoint("node01"));
    EXPECT_FALSE(isDirectConnectEndpoint(""));
}

TEST_F(MetadataDiscoveryTest, MetadataDiscoveryModeFromString) {
    EXPECT_EQ(metadataDiscoveryModeFromString("hybrid"),
              MetadataDiscoveryMode::Hybrid);
    EXPECT_EQ(metadataDiscoveryModeFromString("P2PHANDSHAKE"),
              MetadataDiscoveryMode::P2P);
    EXPECT_EQ(metadataDiscoveryModeFromString("central"),
              MetadataDiscoveryMode::Central);
}

TEST_F(MetadataDiscoveryTest, MetadataDiscoveryModeToString) {
    EXPECT_STREQ(metadataDiscoveryModeToString(MetadataDiscoveryMode::Hybrid),
                 "hybrid");
}
