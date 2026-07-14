// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using Silk.NET.Vulkan;
using SharpEmu.HLE;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace SharpEmu.Libs.VideoOut;

[Flags]
internal enum GuestBufferUsage
{
    None = 0,
    Storage = 1,
    Vertex = 2,
    Index = 4,
}

[Flags]
internal enum GuestBufferAccess
{
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = Read | Write,
}

internal readonly record struct GuestBufferAllocation(
    VkBuffer Buffer,
    DeviceMemory Memory);

internal sealed class GuestBufferResource
{
    public required ulong GuestAddress;
    public required ulong Size;
    public required VkBuffer Buffer;
    public required DeviceMemory Memory;
    public GuestBufferUsage LastUsage;
    public GuestBufferAccess LastAccess;
    public ulong Generation;
    public ulong LastSubmission;
    public int InFlightReferences;
    public required byte[] CpuShadow;
    public required bool[] CpuKnown;
    public List<GuestBufferDirtyRange> GpuDirtyRanges { get; } = [];
    public List<GuestBufferBinding> Bindings { get; } = [];
}

internal sealed record GuestBufferDirtyRange(
    ulong Address,
    ulong Size,
    ICpuMemory GuestMemory);

internal sealed class GuestBufferBinding
{
    public required GuestBufferResource Resource { get; set; }
    public required ulong Offset { get; set; }
    public required ulong Size { get; init; }
    public required GuestBufferUsage Usage { get; init; }
    public required GuestBufferAccess Access { get; init; }
    public ICpuMemory? GuestMemory { get; init; }
}

internal delegate void WriteGuestBuffer(
    GuestBufferResource resource,
    ulong offset,
    ReadOnlySpan<byte> data);

internal delegate void ReadGuestBuffer(
    GuestBufferResource resource,
    ulong offset,
    Span<byte> data);

internal sealed class GuestBufferCache : IDisposable
{
    private const int MinimumRetainedResources = 64;
    private const ulong StaleSubmissionAge = 64;
    private readonly Func<ulong, GuestBufferAllocation> _allocate;
    private readonly Action<GuestBufferResource> _destroy;
    private readonly WriteGuestBuffer _write;
    private readonly ReadGuestBuffer _read;
    private readonly Action _waitForIdle;
    private readonly List<GuestBufferResource> _resources = [];
    private bool _waitingForIdle;

    public GuestBufferCache(
        Func<ulong, GuestBufferAllocation> allocate,
        Action<GuestBufferResource> destroy,
        WriteGuestBuffer write,
        ReadGuestBuffer read,
        Action waitForIdle)
    {
        _allocate = allocate;
        _destroy = destroy;
        _write = write;
        _read = read;
        _waitForIdle = waitForIdle;
    }

    public int Count => _resources.Count;

    public GuestBufferBinding ObtainBuffer(
        ulong address,
        ReadOnlySpan<byte> cpuData,
        GuestBufferUsage usage,
        GuestBufferAccess access,
        ulong submission,
        ICpuMemory? guestMemory = null)
    {
        var size = (ulong)Math.Max(cpuData.Length, sizeof(uint));
        var end = checked(address + size);
        var overlaps = FindOverlaps(address, end);
        GuestBufferResource resource;
        if (overlaps.Count == 1 && Contains(overlaps[0], address, end))
        {
            resource = overlaps[0];
            UploadCpuChanges(resource, address, cpuData);
        }
        else if (overlaps.Count == 0)
        {
            resource = CreateResource(address, size);
            UploadCpuChanges(resource, address, cpuData);
            _resources.Add(resource);
        }
        else
        {
            resource = MergeResources(address, end, cpuData);
        }

        resource.LastUsage = usage;
        resource.LastAccess = access;
        resource.LastSubmission = submission;
        resource.InFlightReferences++;
        var binding = new GuestBufferBinding
        {
            Resource = resource,
            Offset = address - resource.GuestAddress,
            Size = size,
            Usage = usage,
            Access = access,
            GuestMemory = guestMemory,
        };
        resource.Bindings.Add(binding);
        return binding;
    }

    public void Release(GuestBufferBinding binding)
    {
        if (binding.Resource.InFlightReferences <= 0)
        {
            throw new InvalidOperationException("guest buffer reference count underflow");
        }

        binding.Resource.InFlightReferences--;
        binding.Resource.Bindings.Remove(binding);
    }

    public void MarkGpuWritten(GuestBufferBinding binding)
    {
        if ((binding.Access & GuestBufferAccess.Write) == 0 ||
            binding.GuestMemory is not { } guestMemory)
        {
            return;
        }

        AddDirtyRange(
            binding.Resource,
            binding.Resource.GuestAddress + binding.Offset,
            binding.Size,
            guestMemory);
    }

    public bool FlushGpuWrites(ulong address, ulong size)
    {
        var end = checked(address + size);
        var success = true;
        foreach (var resource in _resources)
        {
            var dirtyRanges = resource.GpuDirtyRanges.ToArray();
            foreach (var dirty in dirtyRanges)
            {
                var dirtyEnd = checked(dirty.Address + dirty.Size);
                var flushStart = Math.Max(address, dirty.Address);
                var flushEnd = Math.Min(end, dirtyEnd);
                if (flushStart >= flushEnd)
                {
                    continue;
                }

                var bytes = new byte[checked((int)(flushEnd - flushStart))];
                var resourceOffset = flushStart - resource.GuestAddress;
                _read(resource, resourceOffset, bytes);
                if (!dirty.GuestMemory.TryWrite(flushStart, bytes))
                {
                    success = false;
                    continue;
                }

                RememberCpuBytes(resource, resourceOffset, bytes);
                ReplaceDirtyRange(resource, dirty, flushStart, flushEnd);
            }
        }
        return success;
    }

    public void Collect(ulong completedSubmission)
    {
        if (_waitingForIdle || _resources.Count <= MinimumRetainedResources)
        {
            return;
        }

        for (var index = _resources.Count - 1; index >= 0; index--)
        {
            var resource = _resources[index];
            if (_resources.Count <= MinimumRetainedResources ||
                resource.InFlightReferences != 0 ||
                completedSubmission < resource.LastSubmission ||
                completedSubmission - resource.LastSubmission < StaleSubmissionAge)
            {
                continue;
            }

            FlushResource(resource);
            if (resource.GpuDirtyRanges.Count != 0)
            {
                continue;
            }
            _resources.RemoveAt(index);
            _destroy(resource);
        }
    }

    public void Dispose()
    {
        if (_resources.Any(resource => resource.InFlightReferences != 0))
        {
            WaitForIdle();
        }

        foreach (var resource in _resources)
        {
            if (resource.InFlightReferences != 0)
            {
                throw new InvalidOperationException("guest buffer cache disposed with in-flight resources");
            }
            FlushResource(resource);
            _destroy(resource);
        }
        _resources.Clear();
    }

    private GuestBufferResource MergeResources(
        ulong address,
        ulong end,
        ReadOnlySpan<byte> cpuData)
    {
        WaitForIdle();
        var overlaps = FindOverlaps(address, end);
        var mergedAddress = Math.Min(address, overlaps.Min(resource => resource.GuestAddress));
        var mergedEnd = Math.Max(end, overlaps.Max(ResourceEnd));
        var mergedSize = checked(mergedEnd - mergedAddress);
        var mergedData = new byte[checked((int)mergedSize)];
        var mergedCpuShadow = new byte[mergedData.Length];
        var mergedCpuKnown = new bool[mergedData.Length];
        var cpuChanged = new bool[cpuData.Length];
        ulong generation = 0;
        foreach (var old in overlaps)
        {
            var oldOffset = checked((int)(old.GuestAddress - mergedAddress));
            _read(
                old,
                0,
                mergedData.AsSpan(oldOffset, checked((int)old.Size)));
            old.CpuShadow.CopyTo(mergedCpuShadow, oldOffset);
            old.CpuKnown.CopyTo(mergedCpuKnown, oldOffset);
            generation = Math.Max(generation, old.Generation);
        }

        var cpuOffset = checked((int)(address - mergedAddress));
        for (var index = 0; index < cpuData.Length; index++)
        {
            var mergedIndex = cpuOffset + index;
            if (!mergedCpuKnown[mergedIndex] || mergedCpuShadow[mergedIndex] != cpuData[index])
            {
                mergedData[mergedIndex] = cpuData[index];
                cpuChanged[index] = true;
            }
            mergedCpuShadow[mergedIndex] = cpuData[index];
            mergedCpuKnown[mergedIndex] = true;
        }

        var merged = CreateResource(mergedAddress, mergedSize);
        try
        {
            _write(merged, 0, mergedData);
        }
        catch
        {
            _destroy(merged);
            throw;
        }
        merged.Generation = generation + 1;
        mergedCpuShadow.CopyTo(merged.CpuShadow, 0);
        mergedCpuKnown.CopyTo(merged.CpuKnown, 0);

        foreach (var old in overlaps)
        {
            merged.GpuDirtyRanges.AddRange(old.GpuDirtyRanges);
            foreach (var binding in old.Bindings)
            {
                binding.Resource = merged;
                binding.Offset = checked(old.GuestAddress + binding.Offset - mergedAddress);
                merged.Bindings.Add(binding);
            }
            merged.InFlightReferences += old.InFlightReferences;
            merged.LastSubmission = Math.Max(merged.LastSubmission, old.LastSubmission);
            _resources.Remove(old);
            _destroy(old);
        }
        var changedRunStart = -1;
        for (var index = 0; index <= cpuChanged.Length; index++)
        {
            var changed = index < cpuChanged.Length && cpuChanged[index];
            if (changed && changedRunStart < 0)
            {
                changedRunStart = index;
            }
            else if (!changed && changedRunStart >= 0)
            {
                DiscardGpuDirty(
                    merged,
                    address + checked((ulong)changedRunStart),
                    checked((ulong)(index - changedRunStart)));
                changedRunStart = -1;
            }
        }
        _resources.Add(merged);
        return merged;
    }

    private void UploadCpuChanges(
        GuestBufferResource resource,
        ulong address,
        ReadOnlySpan<byte> cpuData)
    {
        if (cpuData.IsEmpty)
        {
            return;
        }

        var offset = checked((int)(address - resource.GuestAddress));
        var hasChanges = false;
        for (var index = 0; index < cpuData.Length; index++)
        {
            if (!resource.CpuKnown[offset + index] ||
                resource.CpuShadow[offset + index] != cpuData[index])
            {
                hasChanges = true;
                break;
            }
        }
        if (!hasChanges)
        {
            return;
        }

        if (resource.InFlightReferences != 0)
        {
            WaitForIdle();
        }

        var runStart = -1;
        for (var index = 0; index <= cpuData.Length; index++)
        {
            var changed = index < cpuData.Length &&
                (!resource.CpuKnown[offset + index] ||
                 resource.CpuShadow[offset + index] != cpuData[index]);
            if (changed && runStart < 0)
            {
                runStart = index;
            }
            else if (!changed && runStart >= 0)
            {
                DiscardGpuDirty(
                    resource,
                    checked((ulong)(offset + runStart)) + resource.GuestAddress,
                    checked((ulong)(index - runStart)));
                _write(
                    resource,
                    checked((ulong)(offset + runStart)),
                    cpuData[runStart..index]);
                runStart = -1;
            }
        }
        cpuData.CopyTo(resource.CpuShadow.AsSpan(offset));
        Array.Fill(resource.CpuKnown, true, offset, cpuData.Length);
        resource.Generation++;
    }

    private void FlushResource(GuestBufferResource resource)
    {
        foreach (var dirty in resource.GpuDirtyRanges.ToArray())
        {
            _ = FlushGpuWrites(dirty.Address, dirty.Size);
        }
    }

    private void AddDirtyRange(
        GuestBufferResource resource,
        ulong address,
        ulong size,
        ICpuMemory guestMemory)
    {
        var start = address;
        var end = checked(address + size);
        for (var index = resource.GpuDirtyRanges.Count - 1; index >= 0; index--)
        {
            var existing = resource.GpuDirtyRanges[index];
            var existingEnd = checked(existing.Address + existing.Size);
            if (!ReferenceEquals(existing.GuestMemory, guestMemory) ||
                start > existingEnd || existing.Address > end)
            {
                continue;
            }

            start = Math.Min(start, existing.Address);
            end = Math.Max(end, existingEnd);
            resource.GpuDirtyRanges.RemoveAt(index);
        }
        resource.GpuDirtyRanges.Add(new GuestBufferDirtyRange(start, end - start, guestMemory));
    }

    private static void DiscardGpuDirty(
        GuestBufferResource resource,
        ulong address,
        ulong size)
    {
        var end = checked(address + size);
        foreach (var dirty in resource.GpuDirtyRanges.ToArray())
        {
            var dirtyEnd = checked(dirty.Address + dirty.Size);
            if (address >= dirtyEnd || dirty.Address >= end)
            {
                continue;
            }
            ReplaceDirtyRange(resource, dirty, address, end);
        }
    }

    private static void ReplaceDirtyRange(
        GuestBufferResource resource,
        GuestBufferDirtyRange dirty,
        ulong removeStart,
        ulong removeEnd)
    {
        resource.GpuDirtyRanges.Remove(dirty);
        var dirtyEnd = checked(dirty.Address + dirty.Size);
        if (dirty.Address < removeStart)
        {
            resource.GpuDirtyRanges.Add(
                dirty with { Size = removeStart - dirty.Address });
        }
        if (removeEnd < dirtyEnd)
        {
            resource.GpuDirtyRanges.Add(
                dirty with { Address = removeEnd, Size = dirtyEnd - removeEnd });
        }
    }

    private static void RememberCpuBytes(
        GuestBufferResource resource,
        ulong offset,
        ReadOnlySpan<byte> bytes)
    {
        var index = checked((int)offset);
        bytes.CopyTo(resource.CpuShadow.AsSpan(index));
        Array.Fill(resource.CpuKnown, true, index, bytes.Length);
    }

    private GuestBufferResource CreateResource(ulong address, ulong size)
    {
        var allocation = _allocate(size);
        return new GuestBufferResource
        {
            GuestAddress = address,
            Size = size,
            Buffer = allocation.Buffer,
            Memory = allocation.Memory,
            CpuShadow = new byte[checked((int)size)],
            CpuKnown = new bool[checked((int)size)],
            Generation = 1,
        };
    }

    private void WaitForIdle()
    {
        _waitingForIdle = true;
        try
        {
            _waitForIdle();
        }
        finally
        {
            _waitingForIdle = false;
        }
    }

    private List<GuestBufferResource> FindOverlaps(ulong address, ulong end) =>
        _resources
            .Where(resource => address < ResourceEnd(resource) && resource.GuestAddress < end)
            .ToList();

    private static bool Contains(GuestBufferResource resource, ulong address, ulong end) =>
        address >= resource.GuestAddress && end <= ResourceEnd(resource);

    private static ulong ResourceEnd(GuestBufferResource resource) =>
        checked(resource.GuestAddress + resource.Size);

}
