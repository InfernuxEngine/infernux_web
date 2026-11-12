#pragma once

#include "WebGpuRhiDevice.h"
#include <function/renderer/rhi/RhiComputeQueue.h>

#include <deque>

namespace infernux::web
{

/// Bounded numerical submissions on the browser renderer's existing queue.
/// Completion callbacks publish facts; the engine thread retires owners in Collect.
class WebGpuComputeQueue final : public rhi::ComputeQueue
{
  public:
    explicit WebGpuComputeQueue(WebGpuRhiDevice &device, uint32_t slots = 3);
    ~WebGpuComputeQueue() override;

    [[nodiscard]] rhi::SubmissionTicket Submit(const Recorder &record,
                                               rhi::ComputeSubmissionResources resources = {}) override;
    [[nodiscard]] std::optional<rhi::SubmissionTicket> TrySubmit(const Recorder &record,
                                                                 const rhi::ComputeSubmissionResources &resources = {},
                                                                 const Admission &admit = {}) override;
    void Wait(rhi::SubmissionTicket ticket) override;
    [[nodiscard]] bool WaitFor(rhi::SubmissionTicket ticket, uint64_t timeoutNanoseconds) override;
    [[nodiscard]] bool IsComplete(rhi::SubmissionTicket ticket) override;
    void Collect() override;
    bool SetProfilingEnabled(bool enabled) override;
    [[nodiscard]] rhi::GpuTimestampFrame GetProfile() const override;
    [[nodiscard]] uint64_t GetPendingSubmissionCount() const noexcept override;
    [[nodiscard]] uint64_t GetRetainedUploadBytes() const noexcept override;

  private:
    struct Completion;
    [[nodiscard]] rhi::SubmissionTicket Record(const Recorder &record, rhi::ComputeSubmissionResources resources);
    void RequireTicket(rhi::SubmissionTicket ticket) const;

    WebGpuRhiDevice &m_device;
    uint32_t m_capacity;
    rhi::SubmissionSerial m_submitted = 0;
    rhi::SubmissionSerial m_completed = 0;
    std::deque<std::shared_ptr<Completion>> m_pending;
};

} // namespace infernux::web
