import Metal
public enum MetalDevice {
    public static func getDefaultDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.noDeviceFound
        }
        return device
    }
}
public enum MetalError: Error { case noDeviceFound }
