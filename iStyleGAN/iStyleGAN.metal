#include <metal_stdlib>
using namespace metal;

struct FragData
{
    float4 position [[position]];
    float2 textureCoord;
    float size [[point_size]];
};

vertex FragData vertexShader(constant float2* vertexArray [[ buffer(0) ]],
                             uint vertexId [[ vertex_id ]])
{
    FragData out;
    const float2 in = vertexArray[vertexId];
    out.position = float4(in, 0.0, 1.0);
    out.textureCoord = (in + 1.0) * 0.5;
    return out;
}

fragment float4 fragmentShader(FragData fragData [[ stage_in ]],
                               constant float& rms [[ buffer(0) ]],
                               constant float& time [[ buffer(1) ]],
                               texture2d<half, access::sample> tx [[ texture(0) ]])
{
    constexpr sampler txSampler(mag_filter::linear, min_filter::linear, address::mirrored_repeat);
    float2 txPos = fragData.textureCoord * 2.0 - 1.0;
       
    txPos.x *= 2.0;
    txPos.y *= 1.125;
    
    if (length(txPos) >= 1.0) {
        txPos.x *= sin(time * rms) + cos(time + rms);
        txPos.y *= tan(time - rms);
    }

    txPos = txPos * 0.5 + 0.5;
    
    const float third = 1.0 / 3.0;
    const float third2 = 2.0 / 3.0;
    const float txX = txPos.x * third;
    const float txY = third * (1.0 - txPos.y);
    const half r = tx.sample(txSampler, float2(txX, txY)).r;
    const half g = tx.sample(txSampler, float2(txX, txY + third)).r;
    const half b = tx.sample(txSampler, float2(txX, txY + third2)).r;
    float3 tex = float3(r, g, b);
    return float4(tex, 1.0);
}
