#include "WebIndexPacking.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace infernux::web
{
namespace
{
template <typename Index> void RequireAppendCapacity(const std::vector<Index> &target, size_t count, size_t padding = 0)
{
    constexpr size_t maxDrawIndex = std::numeric_limits<uint32_t>::max();
    if (target.size() > maxDrawIndex || count > maxDrawIndex - target.size() ||
        padding > maxDrawIndex - target.size() - count)
        throw std::overflow_error("Web index stream exceeds the DrawIndexed uint32 range");
}
} // namespace

WebPackedIndexRange PackWebIndexRange(WebPackedIndexStreams &streams, MeshIndexFormat requestedFormat,
                                      size_t vertexBase, size_t vertexCount,
                                      const std::vector<uint32_t> &authoritativeIndices, size_t indexStart,
                                      size_t indexCount)
{
    if (vertexBase > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::overflow_error("Web vertex rebase exceeds the DrawIndexed int32 range");
    if (indexStart > authoritativeIndices.size() || indexCount > authoritativeIndices.size() - indexStart)
        throw std::invalid_argument("Web draw index range exceeds the authoritative CPU index stream");
    if (indexCount > static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        throw std::overflow_error("Web draw index count exceeds the DrawIndexed uint32 range");

    const MeshIndexFormat format = ResolveMeshIndexFormat(requestedFormat, vertexCount, authoritativeIndices);
    WebPackedIndexRange range;
    range.format = format;
    range.indexCount = static_cast<uint32_t>(indexCount);
    range.baseVertex = static_cast<int32_t>(vertexBase);
    std::unordered_map<uint32_t, uint32_t> remap;
    remap.reserve(std::min(vertexCount, indexCount));
    std::vector<uint32_t> packedIndices;
    packedIndices.reserve(indexCount);
    range.sourceVertices.reserve(std::min(vertexCount, indexCount));
    for (size_t offset = 0; offset < indexCount; ++offset) {
        const uint32_t sourceIndex = authoritativeIndices[indexStart + offset];
        const auto [entry, inserted] =
            remap.try_emplace(sourceIndex, static_cast<uint32_t>(range.sourceVertices.size()));
        if (inserted)
            range.sourceVertices.push_back(sourceIndex);
        packedIndices.push_back(entry->second);
    }
    if (!range.sourceVertices.empty() &&
        range.sourceVertices.size() - 1 > std::numeric_limits<uint32_t>::max() - vertexBase)
        throw std::overflow_error("Web rebased vertex index exceeds the uint32 GPU range");

    // ResolveMeshIndexFormat validates every authoritative index before this
    // point, including the UInt16 narrowing contract.  Reserve before append
    // so allocation failure also leaves the selected stream unchanged.
    if (format == MeshIndexFormat::UInt16) {
        // WebGPU Queue::WriteBuffer requires a four-byte upload size.  Pad
        // every UInt16 range to an even element count so both the whole stream
        // and every subsequent range offset retain that alignment.  The pad
        // is outside range.indexCount and can never be drawn.
        const size_t padding = indexCount % 2U;
        RequireAppendCapacity(streams.uint16Indices, indexCount, padding);
        range.firstIndex = static_cast<uint32_t>(streams.uint16Indices.size());
        streams.uint16Indices.reserve(streams.uint16Indices.size() + indexCount + padding);
        for (size_t offset = 0; offset < indexCount; ++offset)
            streams.uint16Indices.push_back(static_cast<uint16_t>(packedIndices[offset]));
        if (padding != 0)
            streams.uint16Indices.push_back(0);
    } else if (format == MeshIndexFormat::UInt32) {
        RequireAppendCapacity(streams.uint32Indices, indexCount);
        range.firstIndex = static_cast<uint32_t>(streams.uint32Indices.size());
        streams.uint32Indices.reserve(streams.uint32Indices.size() + indexCount);
        streams.uint32Indices.insert(streams.uint32Indices.end(), packedIndices.begin(), packedIndices.end());
    } else {
        throw std::invalid_argument("Web draw resolved to an unsupported index format");
    }
    return range;
}

} // namespace infernux::web
