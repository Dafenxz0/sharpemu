// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.HLE;
using SharpEmu.Libs.VideoOut;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

var storage = new Dictionary<GuestBufferResource, byte[]>();
ulong nextHandle = 1;
var destroyed = 0;
var waits = 0;

using var cache = new GuestBufferCache(
    size => new GuestBufferAllocation(
        new VkBuffer(nextHandle++),
        new DeviceMemory(nextHandle++)),
    resource =>
    {
        storage.Remove(resource);
        destroyed++;
    },
    (resource, offset, data) =>
    {
        if (!storage.TryGetValue(resource, out var bytes))
        {
            bytes = new byte[checked((int)resource.Size)];
            storage.Add(resource, bytes);
        }
        data.CopyTo(bytes.AsSpan(checked((int)offset)));
    },
    (resource, offset, data) =>
        storage[resource].AsSpan(checked((int)offset), data.Length).CopyTo(data),
    () => waits++);

var original = Words(1, 2, 3, 4);
var first = cache.ObtainBuffer(
    0x1000,
    original,
    GuestBufferUsage.Storage,
    GuestBufferAccess.ReadWrite,
    1);

// Stand in for a compute shader writing the persistent Vulkan allocation.
Words(2, 4, 6, 8).CopyTo(storage[first.Resource], 0);

var second = cache.ObtainBuffer(
    0x1000,
    original,
    GuestBufferUsage.Storage,
    GuestBufferAccess.ReadWrite,
    2);
Assert(storage[first.Resource].SequenceEqual(Words(2, 4, 6, 8)),
    "stale CPU bytes replaced GPU-produced contents");

var contained = cache.ObtainBuffer(
    0x1004,
    original.AsSpan(4, 8),
    GuestBufferUsage.Vertex,
    GuestBufferAccess.Read,
    3);
Assert(storage[first.Resource].SequenceEqual(Words(2, 4, 6, 8)),
    "a stale contained CPU range replaced GPU-produced contents");

var extension = new byte[] { 4, 0, 0, 0, 0x11, 0x22, 0x33, 0x44 };
var overlapping = cache.ObtainBuffer(
    0x100C,
    extension,
    GuestBufferUsage.Vertex,
    GuestBufferAccess.Read,
    4);
Assert(ReferenceEquals(first.Resource, overlapping.Resource),
    "active bindings were not rebound after an overlapping range merge");
Assert(storage[overlapping.Resource].AsSpan(0, 16).SequenceEqual(Words(2, 4, 6, 8)),
    "range merge did not preserve existing GPU data");
Assert(storage[overlapping.Resource].AsSpan(16, 4).SequenceEqual(extension.AsSpan(4)),
    "range merge did not upload the uncovered CPU suffix");
Assert(waits == 1, "overlapping in-flight resources were merged without waiting");

cache.Release(first);
cache.Release(second);
cache.Release(contained);
cache.Release(overlapping);

var cpuOwned = cache.ObtainBuffer(
    0x3000,
    Words(10),
    GuestBufferUsage.Storage,
    GuestBufferAccess.Read,
    5);
var changedCpuData = cache.ObtainBuffer(
    0x3000,
    Words(11),
    GuestBufferUsage.Storage,
    GuestBufferAccess.Read,
    6);
Assert(waits == 2, "CPU mutation did not wait for an in-flight use");
Assert(storage[cpuOwned.Resource].SequenceEqual(Words(11)),
    "changed CPU bytes were not uploaded");
cache.Release(cpuOwned);
cache.Release(changedCpuData);

var guestMemory = new FakeMemory(0x4000, Words(1, 2, 3, 4));
var gpuWritten = cache.ObtainBuffer(
    0x4000,
    Words(1, 2, 3, 4),
    GuestBufferUsage.Storage,
    GuestBufferAccess.ReadWrite,
    7,
    guestMemory);
Words(20, 40, 60, 80).CopyTo(storage[gpuWritten.Resource], 0);
cache.MarkGpuWritten(gpuWritten);
cache.Release(gpuWritten);

Assert(cache.FlushGpuWrites(0x4004, 8), "partial GPU readback failed");
Assert(guestMemory.Bytes.SequenceEqual(Words(1, 40, 60, 4)),
    "partial GPU readback wrote outside the requested range");
Assert(gpuWritten.Resource.GpuDirtyRanges.Count == 2,
    "partial GPU readback did not retain the unflushed dirty ranges");
Assert(cache.FlushGpuWrites(0x4000, 16), "remaining GPU readback failed");
Assert(guestMemory.Bytes.SequenceEqual(Words(20, 40, 60, 80)),
    "GPU readback did not update guest CPU memory");
Assert(gpuWritten.Resource.GpuDirtyRanges.Count == 0,
    "GPU dirty ranges remained after a complete readback");

Words(100, 200, 300, 400).CopyTo(storage[gpuWritten.Resource], 0);
cache.MarkGpuWritten(gpuWritten);
var cpuAfterGpu = Words(7, 40, 60, 80);
Assert(guestMemory.TryWrite(0x4000, cpuAfterGpu), "test CPU write failed");
var cpuOverride = cache.ObtainBuffer(
    0x4000,
    cpuAfterGpu,
    GuestBufferUsage.Storage,
    GuestBufferAccess.Read,
    8,
    guestMemory);
cache.Release(cpuOverride);
Assert(cache.FlushGpuWrites(0x4000, 16), "GPU readback after CPU override failed");
Assert(guestMemory.Bytes.SequenceEqual(Words(7, 200, 300, 400)),
    "GPU readback overwrote a newer CPU-written byte range");

for (ulong index = 0; index < 70; index++)
{
    var binding = cache.ObtainBuffer(
        0x10000 + index * 0x100,
        Words((uint)index),
        GuestBufferUsage.Storage,
        GuestBufferAccess.Read,
        100 + index);
    cache.Release(binding);
}

cache.Collect(500);
Assert(cache.Count == 64, "garbage collection did not retain the expected working set");
Assert(destroyed >= 8, "garbage collection did not destroy stale allocations");

Console.WriteLine("GuestBufferCache persistence, overlap, CPU update, and GC tests passed.");

static byte[] Words(params uint[] values)
{
    var bytes = new byte[values.Length * sizeof(uint)];
    for (var index = 0; index < values.Length; index++)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(
            bytes.AsSpan(index * sizeof(uint)),
            values[index]);
    }
    return bytes;
}

static void Assert(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

sealed class FakeMemory(ulong baseAddress, byte[] bytes) : ICpuMemory
{
    public byte[] Bytes { get; } = bytes;

    public bool TryRead(ulong virtualAddress, Span<byte> destination)
    {
        if (virtualAddress < baseAddress ||
            virtualAddress + (ulong)destination.Length > baseAddress + (ulong)Bytes.Length)
        {
            return false;
        }
        Bytes.AsSpan(checked((int)(virtualAddress - baseAddress)), destination.Length)
            .CopyTo(destination);
        return true;
    }

    public bool TryWrite(ulong virtualAddress, ReadOnlySpan<byte> source)
    {
        if (virtualAddress < baseAddress ||
            virtualAddress + (ulong)source.Length > baseAddress + (ulong)Bytes.Length)
        {
            return false;
        }
        source.CopyTo(Bytes.AsSpan(checked((int)(virtualAddress - baseAddress))));
        return true;
    }
}
