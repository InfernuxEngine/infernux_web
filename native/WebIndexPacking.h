#pragma once

#include <function/renderer/InxRenderStruct.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernux::web
{

/// Draw-local location in one of the two WebGPU index streams.  Keeping the
/// format in the range makes a mesh format change select a different buffer
/// instead of reusing bytes published under the previous interpretation.
struct WebPackedIndexRange
{
    MeshIndexFormat format = MeshIndexFormat::UInt32;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t baseVertex = 0;
    std::vector<uint32_t> sourceVertices;
};

struct WebPackedIndexStreams
{
    std::vector<uint16_t> uint16Indices;
    std::vector<uint32_t> uint32Indices;
};

/// Validate and append one draw to the format-specific GPU stream.
///
/// Referenced vertices are compacted in first-use order. sourceVertices maps
/// the packed vertices to the original mesh; unused submeshes are not uploaded.
/// DrawIndexed(baseVertex) preserves the mesh's index format at any global offset.
/// Validation and capacity reservation complete before either stream changes;
/// malformed ranges and overflows therefore reject the draw atomically.
WebPackedIndexRange PackWebIndexRange(WebPackedIndexStreams &streams, MeshIndexFormat requestedFormat,
                                      size_t vertexBase, size_t vertexCount,
                                      const std::vector<uint32_t> &authoritativeIndices, size_t indexStart,
                                      size_t indexCount);

} // namespace infernux::web
