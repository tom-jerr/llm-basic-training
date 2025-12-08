#include <iostream>
#include <cute/tensor.hpp> // 核心头文件
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace cute;

int main() {
    // 1. 定义一个 Layout
    // 逻辑形状 (Shape): (8行, 32列)
    // 步长 (Stride): (32, 1) -> Row-Major (行优先)
    // 注意：CuTe 默认是列优先，Stride(1, M) 是列优先，Stride(N, 1) 是行优先
    auto layout = make_layout(make_shape(8, 32), make_stride(32, 1));

    // 2. 打印 Layout 信息
    std::cout << "My Layout: " << layout << std::endl;
    print_layout(layout); 

    // 3. 验证坐标映射
    // 看看逻辑坐标 (2, 3) 映射到了哪个物理 offset
    int offset = layout(2, 3); 
    std::cout << "\nCoordinate (2, 3) -> Offset: " << offset << std::endl;

    // 验证：行优先，offset = 2 * 32 + 3 * 1 = 67
    
    // 4. 测试 Swizzle (上一轮对话的内容)
    // 定义 Swizzle<3,3,3>
    auto swizzle = Swizzle<3, 3, 3>{};
    auto layout_swizzled = composition(swizzle, layout);
    
    std::cout << "\nSwizzled Layout: " << layout_swizzled << std::endl;
    // 看看 Swizzle 后的 offset
    int offset_swizzled = layout_swizzled(2, 3);
    std::cout << "Coordinate (2, 3) -> Swizzled Offset: " << offset_swizzled << std::endl;

    return 0;
}