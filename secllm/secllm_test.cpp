#include "secllm.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
  constexpr int num_iter = 1000000;
  
  std::vector<int> cnt(256, 0);

  for (int i = 0; i < num_iter; ++i) {
    auto cprng = GenerateCPRNG();
    
    int byte1 = cprng & 0xFF;
    int byte2 = (cprng >> 8) & 0xFF;
    int byte3 = (cprng >> 16) & 0xFF;
    int byte4 = (cprng >> 24) & 0xFF;

    cnt.at(byte1)++;
    cnt.at(byte2)++;
    cnt.at(byte3)++;
    cnt.at(byte4)++;
  }

  // Compute stdev of the counts
  double mean = 0.0;
  for (int i = 0; i < 256; ++i) {
    mean += cnt.at(i);
  }
  mean /= 256;

  double stdev = 0.0;
  for (int i = 0; i < 256; ++i) {
    stdev += (cnt.at(i) - mean) * (cnt.at(i) - mean);
  }
  stdev = std::sqrt(stdev / 256);
/*
  for (int i = 0; i < 256; ++i) {
    std::cout << i << ": " << cnt.at(i) << std::endl;
  }
*/
  std::cout << "Mean: " << mean << std::endl;
  std::cout << "Stdev: " << stdev << std::endl;

  // measure time 
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_iter; ++i) {
    auto cprng = GenerateCPRNG();
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Generation time per cprng: " << static_cast<float>(elapsed.count() / num_iter) << " s" << std::endl;

  return 0;
}