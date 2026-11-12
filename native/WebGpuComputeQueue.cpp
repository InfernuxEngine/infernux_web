#include "WebGpuComputeQueue.h"

#include <chrono>
#include <emscripten.h>
#include <limits>
#include <stdexcept>

namespace infernux::web
{
namespace
{
class RecordingContext final : public rhi::ComputeRecordingContext
{
  public:
    RecordingContext(WebGpuRhiDevice &device, rhi::SubmissionSerial submitted)
        : m_device(device), m_submitted(submitted), m_encoder(device.NativeDevice().CreateCommandEncoder())
    {
    }
    ~RecordingContext() override
    {
        EndCompute();
    }
    rhi::ComputeCommandEncoder Compute() override
    {
        static const rhi::ComputeCommandEncoder::DispatchTable dispatch = {
            [](void *raw, rhi::ComputePipelineHandle pipeline) { From(raw).Encoder().BindPipeline(pipeline); },
            [](void *raw, rhi::ComputePipelineHandle pipeline, uint32_t set, rhi::BindGroupHandle group) {
                From(raw).Encoder().BindGroup(pipeline, set, group);
            },
            [](void *raw, rhi::ComputePipelineHandle pipeline, uint32_t bytes, const void *data) {
                From(raw).Encoder().PushConstants(pipeline, bytes, data);
            },
            [](void *raw, uint32_t x, uint32_t y, uint32_t z) { From(raw).Encoder().Dispatch(x, y, z); },
            [](void *raw, rhi::BufferHandle arguments, uint64_t offset) {
                From(raw).Encoder().DispatchIndirect(arguments, offset);
            },
            [](void *raw, rhi::ComputePipelineHandle pipeline, rhi::BindGroupHandle group) {
                return From(raw).Encoder().CanRecord(pipeline, group);
            },
        };
        return {this, &dispatch};
    }
    rhi::TransferCommandEncoder Transfer() override
    {
        EndCompute();
        if (!m_transfer.encoder)
            m_transferEncoder = m_device.MakeTransferCommandEncoder(m_transfer, m_encoder);
        return m_transferEncoder;
    }
    void Require(rhi::SubmissionTicket producer) override
    {
        if (!producer.IsValid())
            return;
        if (producer.device != m_device.GetDeviceId() ||
            (producer.queue == rhi::QueueRole::Compute && producer.serial > m_submitted))
            throw std::invalid_argument("WebGPU compute requires an accepted same-device producer");
        // The browser exposes one ordered queue for graphics/compute/copies.
        // Every producer has already been submitted; no host wait is required.
    }
    void PipelineBarrier(rhi::PipelineStage, rhi::Access, rhi::PipelineStage, rhi::Access) override
    {
        // End the current usage scope before a later transfer/compute scope.
        EndCompute();
    }
    wgpu::CommandBuffer Finish()
    {
        EndCompute();
        return m_encoder.Finish();
    }
    uint64_t UploadStagingBytes() const noexcept
    {
        return m_transfer.uploadStagingBytes;
    }
    uint64_t UploadAllocations() const noexcept
    {
        return m_transfer.uploadAllocations;
    }

  private:
    static RecordingContext &From(void *raw)
    {
        return *static_cast<RecordingContext *>(raw);
    }
    const rhi::ComputeCommandEncoder &Encoder()
    {
        if (!m_compute.pass)
            m_computeEncoder = m_device.MakeComputeCommandEncoder(m_compute, m_encoder.BeginComputePass());
        return m_computeEncoder;
    }
    void EndCompute()
    {
        if (m_compute.pass) {
            m_compute.pass.End();
            m_compute = {};
        }
    }
    WebGpuRhiDevice &m_device;
    rhi::SubmissionSerial m_submitted;
    wgpu::CommandEncoder m_encoder;
    WebGpuComputeCommandContext m_compute;
    rhi::ComputeCommandEncoder m_computeEncoder;
    WebGpuTransferCommandContext m_transfer;
    rhi::TransferCommandEncoder m_transferEncoder;
};
} // namespace

struct WebGpuComputeQueue::Completion
{
    rhi::SubmissionTicket ticket;
    rhi::ComputeSubmissionResources resources;
    wgpu::QueueWorkDoneStatus status = wgpu::QueueWorkDoneStatus::Error;
    bool commandsDone = false;
    bool validationDone = false;
    std::string error;
    uint64_t uploadStagingBytes = 0;

    bool Ready() const noexcept
    {
        return commandsDone && validationDone;
    }
};

WebGpuComputeQueue::WebGpuComputeQueue(WebGpuRhiDevice &device, uint32_t slots) : m_device(device), m_capacity(slots)
{
    if (!slots)
        throw std::invalid_argument("WebGPU compute requires a nonempty submission pool");
}

WebGpuComputeQueue::~WebGpuComputeQueue()
{
    // Callbacks retain accepted resources independently of the queue wrapper.
    // Normal engine shutdown drains first; a lost device may complete with error.
    for (const auto &pending : m_pending)
        while (!pending->Ready())
            emscripten_sleep(0);
}

rhi::SubmissionTicket WebGpuComputeQueue::Submit(const Recorder &record, rhi::ComputeSubmissionResources resources)
{
    Collect();
    if (m_pending.size() == m_capacity)
        Wait(m_pending.front()->ticket);
    return Record(record, std::move(resources));
}

std::optional<rhi::SubmissionTicket> WebGpuComputeQueue::TrySubmit(const Recorder &record,
                                                                   const rhi::ComputeSubmissionResources &resources,
                                                                   const Admission &admit)
{
    Collect();
    if (m_pending.size() == m_capacity || (admit && !admit()))
        return std::nullopt;
    return Record(record, resources);
}

rhi::SubmissionTicket WebGpuComputeQueue::Record(const Recorder &record, rhi::ComputeSubmissionResources resources)
{
    const auto begin = std::chrono::steady_clock::now();
    auto completion = std::make_shared<Completion>();
    completion->ticket = {m_device.GetDeviceId(), rhi::QueueRole::Compute, m_submitted + 1};
    completion->resources = std::move(resources);
    // WebGPU reports command validation asynchronously. Command completion
    // alone must not publish a failed packet as a successful compute value.
    struct ValidationScope
    {
        wgpu::Device device;
        std::shared_ptr<Completion> completion;
        ~ValidationScope()
        {
            device.PopErrorScope(
                wgpu::CallbackMode::AllowSpontaneous,
                [value = completion](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message) {
                    if (status != wgpu::PopErrorScopeStatus::Success)
                        value->error = "WebGPU compute validation could not complete";
                    else if (type != wgpu::ErrorType::NoError)
                        value->error.assign(message.data, message.length);
                    value->validationDone = true;
                });
        }
    } validation{m_device.NativeDevice(), completion};
    validation.device.PushErrorScope(wgpu::ErrorFilter::Validation);
    m_device.ClearError();
    RecordingContext context(m_device, m_submitted);
    if (!record(context))
        throw std::runtime_error("WebGPU compute recording failed");
    const auto commands = context.Finish();
    if (!m_device.LastError().empty())
        throw std::runtime_error(m_device.LastError());
    completion->uploadStagingBytes = context.UploadStagingBytes();
    m_pending.push_back(completion);
    m_device.NativeQueue().Submit(1, &commands);
    m_submitted = completion->ticket.serial;
    m_device.NativeQueue().OnSubmittedWorkDone(wgpu::CallbackMode::AllowSpontaneous,
                                               [completion](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
                                                   completion->status = status;
                                                   completion->commandsDone = true;
                                               });
    for (uint64_t index = 0; index < context.UploadAllocations(); ++index) {
        RecordStagingAllocation();
        RecordHostMap();
    }
    RecordSubmission(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
    return completion->ticket;
}

void WebGpuComputeQueue::RequireTicket(rhi::SubmissionTicket ticket) const
{
    if (ticket.device != m_device.GetDeviceId() || ticket.queue != rhi::QueueRole::Compute || !ticket.IsValid() ||
        ticket.serial > m_submitted)
        throw std::invalid_argument("Compute completion does not belong to this WebGPU queue");
}

void WebGpuComputeQueue::Wait(rhi::SubmissionTicket ticket)
{
    (void)WaitFor(ticket, std::numeric_limits<uint64_t>::max());
}

bool WebGpuComputeQueue::WaitFor(rhi::SubmissionTicket ticket, uint64_t timeoutNanoseconds)
{
    RequireTicket(ticket);
    Collect();
    if (ticket.serial <= m_completed)
        return true;
    const auto begin = std::chrono::steady_clock::now();
    while (ticket.serial > m_completed) {
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        if (timeoutNanoseconds != std::numeric_limits<uint64_t>::max() &&
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()) >=
                timeoutNanoseconds) {
            RecordWait(std::chrono::duration<double, std::milli>(elapsed).count());
            return false;
        }
        // JSPI yields the suspended caller to browser completion events.
        // TrySubmit/Collect never sleep; only an explicit blocking API does.
        emscripten_sleep(0);
        Collect();
    }
    RecordWait(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
    return true;
}

bool WebGpuComputeQueue::IsComplete(rhi::SubmissionTicket ticket)
{
    RequireTicket(ticket);
    Collect();
    return ticket.serial <= m_completed;
}

void WebGpuComputeQueue::Collect()
{
    while (!m_pending.empty() && m_pending.front()->Ready()) {
        const auto &completion = m_pending.front();
        if (!completion->error.empty())
            throw std::runtime_error(completion->error);
        if (completion->status != wgpu::QueueWorkDoneStatus::Success)
            throw std::runtime_error("WebGPU compute submission failed");
        m_completed = completion->ticket.serial;
        if (completion->resources.receipt)
            completion->resources.receipt->complete = true;
        // Publish the completion before releasing owners with older tickets.
        m_pending.pop_front();
    }
}

bool WebGpuComputeQueue::SetProfilingEnabled(bool enabled)
{
    // Timestamp-query must be negotiated with the device. Do not report host
    // submit/wait time as device execution time on the baseline adapter.
    return !enabled;
}

rhi::GpuTimestampFrame WebGpuComputeQueue::GetProfile() const
{
    return {};
}

uint64_t WebGpuComputeQueue::GetPendingSubmissionCount() const noexcept
{
    return m_pending.size();
}

uint64_t WebGpuComputeQueue::GetRetainedUploadBytes() const noexcept
{
    uint64_t bytes = 0;
    for (const auto &completion : m_pending)
        bytes += completion->uploadStagingBytes;
    return bytes;
}

} // namespace infernux::web
