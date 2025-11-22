import Foundation

let vocab = 50257
let dModel = 2304
let dState = 16
let dConv = 4
let dInner = dModel * 2
let dtRank = dModel / 16
let dFF = 9216
let numExperts = 8

let embParams = vocab * dModel
print("Embedding: \(embParams / 1_000_000)M")

let mambaParams = (
    dModel * (2 * dInner) +
    dInner * dConv +
    dInner * (dtRank + 2*dState) +
    dtRank * dInner +
    dInner * dState +
    dInner +
    dInner * dModel +
    dModel
)
print("Per Mamba: \(mambaParams / 1_000_000)M")
print("21 Mamba: \(21 * mambaParams / 1_000_000)M")

let expertParams = 2 * dModel * dFF
let moeParams = dModel * numExperts + numExperts * expertParams + dModel
print("Per MoE: \(moeParams / 1_000_000)M")
print("7 MoE: \(7 * moeParams / 1_000_000)M")

let total = embParams + 21 * mambaParams + 7 * moeParams
print("\nTOTAL: \(total / 1_000_000)M params")
