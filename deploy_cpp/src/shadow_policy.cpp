// Shadow-only TorchScript check. It intentionally has no LowCmd publisher,
// no LocoClient and no user-control switch. Build only after LibTorch is
// installed on the target Ubuntu machine.
#include <torch/script.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: g1_shadow_policy <locked_elbow_main_84obs_14act_policy.pt>\n";
    return 2;
  }
  auto policy = torch::jit::load(argv[1], torch::kCPU);
  policy.eval();
  auto observation = torch::zeros({1, 84}, torch::TensorOptions().dtype(torch::kFloat32));
  auto action = policy.forward({observation}).toTensor();
  if (action.sizes() != torch::IntArrayRef({1, 14}))
    throw std::runtime_error("Expected policy output [1,14].");
  std::cout << "SHADOW POLICY PASS: input=[1,84], output=[1,14]. No robot connection or command publisher exists.\n";
}
