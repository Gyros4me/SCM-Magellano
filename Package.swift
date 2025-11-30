// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SCMMagellano",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MagellanoCore", targets: ["MagellanoCore"]),
        .library(name: "MagellanoMetal", targets: ["MagellanoMetal"]),
        .executable(name: "MagellanoCLI", targets: ["MagellanoCLI"])
    ],
    targets: [
        .target(
            name: "MagellanoCore",
            dependencies: ["MagellanoMetal"],
            resources: [
                .process("Kernels/SelectiveScan.metal"),
                .process("Kernels/ElementwiseKernels.metal"),
                .process("Resources/NF4Kernels.metal")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"])
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate")
            ]
        ),
        .target(
            name: "MagellanoMetal",
            resources: [
                .process("Kernels/MoE.metal")
            ],
            linkerSettings: [
                .linkedFramework("Metal")
            ]
        ),
        .executableTarget(
            name: "MagellanoCLI",
            dependencies: ["MagellanoCore"]
        )
    ]
)
