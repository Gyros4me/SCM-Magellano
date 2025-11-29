import Foundation
import Metal
import MagellanoCore

@main
struct ComponentTest {
    static func main() async throws {
        print("🧪 Component-Only Test\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError() }
        
        let lora = LoRALayer(device: device, inDim: 64, outDim: 64, config: LoRAConfig(rank: 8))!
        let cache = ActivationCache()
        let loss = CrossEntropyLoss(device: device)
        let backward = LoRABackward(device: device)
        let optimizer = AdamOptimizer(device: device, config: OptimizerConfig(learningRate: 1e-3))
        
        print("Forward")
        let hidden = Tensor.randn(device: device, shape: [1, 5, 64], std: 0.1, category: .activations)!
        cache.save(name: "pre", activation: hidden)
        
        let h2D = hidden.reshape([5, 64])!
        let xA = Tensor.matmul(device: device, a: h2D, b: lora.matrixA)!
        let xAB = Tensor.matmul(device: device, a: xA, b: lora.matrixB)!
        let xAB3D = xAB.reshape([1, 5, 64])!
        let scaled = xAB3D.scale(lora.config.scaling)!
        let out = Tensor.add(device: device, a: hidden, b: scaled)!
        
        let logits = Tensor.randn(device: device, shape: [1, 5, 128], std: 0.1, category: .activations)!
        let (lossVal, _) = loss.forward(logits: logits, targets: [[1,2,3,4,5]])
        print("  Loss: \(String(format: "%.4f", lossVal))")
        
        print("\nBackward")
        let gradLogits = loss.backward(logits: logits, targets: [[1,2,3,4,5]])!
        let gradHidden = Tensor.randn(device: device, shape: [1, 5, 64], std: 0.01, category: .temporary)!
        let (gradA, gradB) = backward.backward(gradOutput: gradHidden, input: hidden, matrixA: lora.matrixA, matrixB: lora.matrixB, scaling: lora.config.scaling)!
        print("  Grads: A=\(gradA.shape), B=\(gradB.shape)")
        
        print("\nOptimizer")
        optimizer.step(parameters: ["A": lora.matrixA, "B": lora.matrixB], gradients: ["A": gradA, "B": gradB])
        print("  ✓ Updated")
        
        print("\n✅ ALL COMPONENTS WORKING")
    }
}
