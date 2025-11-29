// Sources/MagellanoCore/Data/DataLoader.swift
import Foundation
import Metal

public struct DataConfig: Codable, Sendable {
    public let batchSize: Int
    public let seqLength: Int
    public let vocabSize: Int
    public let shuffleData: Bool
    public let numWorkers: Int
    
    public init(batchSize: Int = 2, seqLength: Int = 128, vocabSize: Int = 50257, shuffleData: Bool = true, numWorkers: Int = 2) {
        self.batchSize = batchSize
        self.seqLength = seqLength
        self.vocabSize = vocabSize
        self.shuffleData = shuffleData
        self.numWorkers = numWorkers
    }
}

public struct Batch: Sendable {
    public let inputIds: [[Int]]      // [batchSize, seqLength]
    public let targetIds: [[Int]]     // [batchSize, seqLength]
    public let attentionMask: [[Int]] // [batchSize, seqLength]
}

public final class DataLoader: @unchecked Sendable {
    private let config: DataConfig
    private var data: [String]
    private var currentIndex: Int = 0
    private let tokenizer: SimpleTokenizer
    
    public init(config: DataConfig, dataPath: String? = nil) {
        self.config = config
        self.tokenizer = SimpleTokenizer(vocabSize: config.vocabSize)
        
        // Load data from file or use sample
        if let path = dataPath, let content = try? String(contentsOfFile: path) {
            self.data = content.components(separatedBy: "\n").filter { !$0.isEmpty }
        } else {
            // Sample educational dataset
            self.data = Self.sampleEducationalData()
        }
        
        if config.shuffleData {
            self.data.shuffle()
        }
    }
    
    public func nextBatch() -> Batch? {
        guard currentIndex < data.count else {
            currentIndex = 0
            if config.shuffleData {
                data.shuffle()
            }
            return nil  // Epoch complete
        }
        
        var inputIds: [[Int]] = []
        var targetIds: [[Int]] = []
        var attentionMask: [[Int]] = []
        
        for _ in 0..<config.batchSize {
            guard currentIndex < data.count else { break }
            
            let text = data[currentIndex]
            currentIndex += 1
            
            // Tokenize
            let tokens = tokenizer.encode(text)
            
            // Create sequences
            if tokens.count < config.seqLength + 1 {
                // Pad short sequences
                var input = Array(tokens.prefix(config.seqLength))
                var target = Array(tokens.dropFirst().prefix(config.seqLength))
                
                while input.count < config.seqLength {
                    input.append(0)  // PAD token
                    target.append(0)
                }
                
                inputIds.append(input)
                targetIds.append(target)
                attentionMask.append(Array(repeating: 1, count: tokens.count) + Array(repeating: 0, count: config.seqLength - tokens.count))
            } else {
                // Use sliding window for long sequences
                let start = Int.random(in: 0...(tokens.count - config.seqLength - 1))
                let input = Array(tokens[start..<(start + config.seqLength)])
                let target = Array(tokens[(start + 1)..<(start + config.seqLength + 1)])
                
                inputIds.append(input)
                targetIds.append(target)
                attentionMask.append(Array(repeating: 1, count: config.seqLength))
            }
        }
        
        return Batch(inputIds: inputIds, targetIds: targetIds, attentionMask: attentionMask)
    }
    
    public func reset() {
        currentIndex = 0
        if config.shuffleData {
            data.shuffle()
        }
    }
    
    public var hasMoreBatches: Bool {
        currentIndex < data.count
    }
    
    private static func sampleEducationalData() -> [String] {
        return [
            "The Mamba architecture provides linear time complexity for sequence modeling.",
            "Mixture of Experts enables sparse activation of neural network capacity.",
            "QLoRA quantization reduces memory requirements while maintaining model quality.",
            "Apple Silicon's unified memory architecture enables efficient ML inference.",
            "Metal Performance Shaders provide GPU acceleration for matrix operations.",
            "Gradient checkpointing trades computation for memory in deep learning.",
            "Low-rank adaptation adds trainable parameters to frozen pretrained models.",
            "Selective state space models excel at long-range dependency modeling.",
            "NF4 quantization uses non-uniform bins optimized for normal distributions.",
            "Load balancing loss encourages equal expert utilization in MoE layers."
        ]
    }
}

// Simple tokenizer (character-level for demo, use GPT-2 tokenizer for production)
public final class SimpleTokenizer: Sendable {
    private let vocabSize: Int
    private let vocab: [String: Int]
    private let invVocab: [Int: String]
    
    public init(vocabSize: Int) {
        self.vocabSize = vocabSize
        
        // Build character vocabulary
        var vocab: [String: Int] = ["<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3]
        var invVocab: [Int: String] = [0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"]
        
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-'"
        for (idx, char) in chars.enumerated() {
            let token = String(char)
            vocab[token] = idx + 4
            invVocab[idx + 4] = token
        }
        
        self.vocab = vocab
        self.invVocab = invVocab
    }
    
    public func encode(_ text: String) -> [Int] {
        var tokens = [vocab["<BOS>"]!]
        for char in text {
            let token = String(char)
            tokens.append(vocab[token] ?? vocab["<UNK>"]!)
        }
        tokens.append(vocab["<EOS>"]!)
        return tokens
    }
    
    public func decode(_ tokens: [Int]) -> String {
        tokens.compactMap { invVocab[$0] }
            .filter { $0 != "<PAD>" && $0 != "<BOS>" && $0 != "<EOS>" }
            .joined()
    }
}
