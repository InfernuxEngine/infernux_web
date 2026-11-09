#pragma once

#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxTexture/TextureReadTask.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace infernux::web
{

// Drive the registry's owner-thread tickets using the browser's inline job queue.
// Keep staged bytes private until the cursor validates the complete artifact.
class WebTextureAssetLoad final
{
  public:
    enum class Result
    {
        Pending,
        Ready,
        Settled,
        Missing,
    };

    struct Data
    {
        TextureCpuData texture;
        std::string filterMode;
        std::string wrapMode;
        int anisoLevel = 1;
    };

    Result Poll(const std::string &guid)
    {
        auto &registry = AssetRegistry::Instance();
        const auto *database = registry.GetAssetDatabase();
        if (!database)
            return Result::Pending;

        const auto contentGeneration = registry.GetAssetContentGeneration(guid);
        const auto runtimeVersion = registry.GetAssetVersion(guid);
        if (m_contentGeneration != contentGeneration || m_runtimeVersion != runtimeVersion) {
            ResetWork();
            m_contentGeneration = contentGeneration;
            m_runtimeVersion = runtimeVersion;
            m_settled = false;
        }
        if (database->IsStagingRuntimeArtifact(guid, ResourceType::Texture))
            return Result::Pending;
        if (!database->ContainsGuid(guid)) {
            ResetWork();
            m_settled = false;
            return Result::Missing;
        }
        if (m_settled)
            return Result::Settled;

        try {
            if (!m_read) {
                auto texture = registry.GetAsset<InxTexture>(guid);
                if (!texture) {
                    if (!m_metadata)
                        m_metadata = registry.BeginLoadAsset(guid, ResourceType::Texture);
                    if (!registry.TrySubmitAssetLoad(m_metadata) || !registry.TryCommitAssetLoad(m_metadata))
                        return Result::Pending;
                    texture = registry.GetAsset<InxTexture>(guid);
                    m_metadata.reset();
                    if (!texture)
                        return Result::Pending;
                }
                m_metadata.reset();
                m_runtimeVersion = registry.GetAssetVersion(guid);
                m_data.filterMode = texture->GetFilterMode();
                m_data.wrapMode = texture->GetWrapMode();
                m_data.anisoLevel = texture->GetAnisoLevel();
                m_read = registry.BeginTextureRead(guid);
            }
            if (m_read->IsPending())
                return Result::Pending;

            const auto state = m_read->GetState();
            if (state == TextureReadState::Output) {
                auto &texture = m_data.texture;
                if (texture.mipLevels.empty()) {
                    const auto &description = m_read->GetDescription();
                    texture.dimension = description.dimension;
                    texture.semantic = description.semantic;
                    texture.format = description.format;
                    texture.mipLevels = description.mipLevels;
                    texture.bakeBasis = description.bakeBasis;
                    texture.valueMin = description.valueMin;
                    texture.valueMax = description.valueMax;
                    texture.bytes.reserve(static_cast<size_t>(description.payloadBytes));
                }
                const auto &output = m_read->GetOutput();
                texture.bytes.insert(texture.bytes.end(), output.begin(), output.end());
                m_read->ConsumeOutput();
            } else if (state == TextureReadState::Complete) {
                m_read.reset();
                m_settled = true;
                return Result::Ready;
            }

            m_read->Advance();
            return Result::Pending;
        } catch (const AssetLoadSuperseded &) {
            ResetWork();
            return Result::Pending;
        } catch (...) {
            ResetWork();
            m_settled = true;
            throw;
        }
    }

    [[nodiscard]] Data TakeData()
    {
        return std::exchange(m_data, {});
    }

  private:
    void ResetWork()
    {
        m_metadata.reset();
        m_read.reset();
        m_data = {};
    }

    std::shared_ptr<AssetLoadTicket> m_metadata;
    std::shared_ptr<TextureReadTask> m_read;
    Data m_data;
    uint64_t m_runtimeVersion = 0;
    uint64_t m_contentGeneration = 0;
    bool m_settled = false;
};

} // namespace infernux::web
