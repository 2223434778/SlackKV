// Phase 0 transfer microbenchmark. Timings use CUDA events; allocation is outside timed regions.
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

struct Options { std::string output, p2p_output; std::vector<size_t> sizes; int warmup = 10, iterations = 30; };
struct Direction { std::string name; int src, dst; bool peer; bool h2d; bool d2h; };

__global__ void fill_pattern(unsigned char* ptr, size_t n) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) ptr[i] = static_cast<unsigned char>((i * 131ULL + 17ULL) & 0xffULL);
}

static unsigned char pattern(size_t i) { return static_cast<unsigned char>((i * 131ULL + 17ULL) & 0xffULL); }
static void csvq(std::ostream& os, const std::string& s) { os << '"'; for (char c : s) os << (c == '"' ? "\"\"" : std::string(1, c)); os << '"'; }
static std::vector<size_t> default_sizes() { return {16ULL<<20,64ULL<<20,256ULL<<20,512ULL<<20,1ULL<<30,2ULL<<30,4ULL<<30,8ULL<<30}; }
static std::vector<size_t> parse_sizes(const std::string& input) {
  std::vector<size_t> out; std::stringstream ss(input); std::string v;
  while (std::getline(ss, v, ',')) out.push_back(static_cast<size_t>(std::stoull(v)) << 20);
  return out;
}
static Options parse(int argc, char** argv) {
  Options o; o.sizes = default_sizes();
  for (int i=1;i<argc;i++) { std::string a(argv[i]);
    if (a=="--output" && i+1<argc) o.output=argv[++i];
    else if (a=="--p2p-output" && i+1<argc) o.p2p_output=argv[++i];
    else if (a=="--sizes-mib" && i+1<argc) o.sizes=parse_sizes(argv[++i]);
    else if (a=="--warmup" && i+1<argc) o.warmup=std::stoi(argv[++i]);
    else if (a=="--iterations" && i+1<argc) o.iterations=std::stoi(argv[++i]);
    else throw std::runtime_error("unknown or incomplete option: " + a);
  }
  if (o.output.empty()) throw std::runtime_error("--output is required");
  return o;
}

static bool peer_access(int src, int dst) { int can=0; CUDA_OK(cudaDeviceCanAccessPeer(&can, src, dst)); return can != 0; }
static void write_p2p(const std::string& path, bool p01, bool p10) {
  if (path.empty()) return;
  std::ofstream f(path);
  f << "{\n  \"devices\": [0, 1],\n  \"can_access_peer\": [[true, " << (p01?"true":"false")
    << "], [" << (p10?"true":"false") << ", true]],\n  \"method\": \"cudaDeviceCanAccessPeer\"\n}\n";
}
static bool check_device(unsigned char* device, int device_id, size_t n) {
  const size_t points[3] = {0, n/2, n-1}; unsigned char got[3]{};
  CUDA_OK(cudaSetDevice(device_id));
  for (int i=0;i<3;i++) CUDA_OK(cudaMemcpy(&got[i], device + points[i], 1, cudaMemcpyDeviceToHost));
  for (int i=0;i<3;i++) if (got[i] != pattern(points[i])) return false;
  return true;
}
static void write_row(std::ofstream& out, const Direction& d, size_t bytes, int iteration, float ms,
                      bool p2p, bool correct, bool success, const std::string& error) {
  out << "native_cuda," << d.name << ',' << d.src << ',' << d.dst << ',' << bytes << ','
      << std::fixed << std::setprecision(3) << (double(bytes)/(1ULL<<20)) << ',' << iteration << ',';
  if (success) out << std::setprecision(6) << ms << ',' << std::setprecision(6) << (double(bytes)/(double(ms)*1e6));
  else out << ",";
  out << ',' << (p2p?"true":"false") << ',' << (correct?"true":"false") << ',' << (success?"true":"false") << ',';
  csvq(out,error); out << '\n';
}

static void run_one(const Direction& d, size_t bytes, const Options& opt, bool p2p_ok, std::ofstream& out) {
  if (d.peer && !p2p_ok) { write_row(out,d,bytes,0,0,false,false,false,"direct CUDA peer access unavailable"); return; }
  unsigned char *src_dev=nullptr, *dst_dev=nullptr; unsigned char* host=nullptr; cudaStream_t stream=nullptr; cudaEvent_t start=nullptr,end=nullptr;
  try {
    if (d.peer) { CUDA_OK(cudaSetDevice(d.src)); CUDA_OK(cudaMalloc(&src_dev, bytes)); CUDA_OK(cudaSetDevice(d.dst)); CUDA_OK(cudaMalloc(&dst_dev, bytes)); }
    else if (d.h2d) { CUDA_OK(cudaSetDevice(d.dst)); CUDA_OK(cudaMalloc(&dst_dev, bytes)); CUDA_OK(cudaMallocHost(&host, bytes)); }
    else { CUDA_OK(cudaSetDevice(d.src)); CUDA_OK(cudaMalloc(&src_dev, bytes)); CUDA_OK(cudaMallocHost(&host, bytes)); }
    int timing_dev = d.peer ? d.src : (d.h2d ? d.dst : d.src); CUDA_OK(cudaSetDevice(timing_dev));
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&end));
    if (d.peer || d.d2h) { CUDA_OK(cudaSetDevice(d.src)); fill_pattern<<<(bytes+255)/256,256>>>(src_dev,bytes); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); }
    if (d.h2d) for (size_t i=0;i<bytes;i++) host[i]=pattern(i);
    auto copy = [&]() { if (d.peer) { CUDA_OK(cudaSetDevice(d.src)); CUDA_OK(cudaMemcpyPeerAsync(dst_dev,d.dst,src_dev,d.src,bytes,stream)); }
                         else if (d.h2d) { CUDA_OK(cudaSetDevice(d.dst)); CUDA_OK(cudaMemcpyAsync(dst_dev,host,bytes,cudaMemcpyHostToDevice,stream)); }
                         else { CUDA_OK(cudaSetDevice(d.src)); CUDA_OK(cudaMemcpyAsync(host,src_dev,bytes,cudaMemcpyDeviceToHost,stream)); } };
    for (int i=0;i<opt.warmup;i++) { copy(); CUDA_OK(cudaStreamSynchronize(stream)); }
    bool correct = d.peer ? check_device(dst_dev,d.dst,bytes) : (d.h2d ? check_device(dst_dev,d.dst,bytes) :
      (host[0]==pattern(0) && host[bytes/2]==pattern(bytes/2) && host[bytes-1]==pattern(bytes-1)));
    if (!correct) { write_row(out,d,bytes,0,0,p2p_ok,false,false,"correctness validation failed"); }
    else for (int i=0;i<opt.iterations;i++) { CUDA_OK(cudaSetDevice(timing_dev)); CUDA_OK(cudaEventRecord(start,stream)); copy(); CUDA_OK(cudaEventRecord(end,stream)); CUDA_OK(cudaEventSynchronize(end)); float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,start,end)); write_row(out,d,bytes,i,ms,p2p_ok,true,true,""); }
  } catch (const std::exception& e) { write_row(out,d,bytes,0,0,p2p_ok,false,false,e.what()); }
  if (start) cudaEventDestroy(start); if (end) cudaEventDestroy(end); if (stream) cudaStreamDestroy(stream);
  if (src_dev) { cudaSetDevice(d.src); cudaFree(src_dev); } if (dst_dev) { cudaSetDevice(d.dst); cudaFree(dst_dev); } if (host) cudaFreeHost(host);
}

int main(int argc, char** argv) {
  try {
    Options opt=parse(argc,argv); int count=0; CUDA_OK(cudaGetDeviceCount(&count)); if (count<2) throw std::runtime_error("Phase 0 requires at least two CUDA GPUs");
    bool p01=peer_access(0,1), p10=peer_access(1,0); write_p2p(opt.p2p_output,p01,p10);
    if (p01) { CUDA_OK(cudaSetDevice(0)); cudaError_t peer_error=cudaDeviceEnablePeerAccess(1,0); if(peer_error!=cudaSuccess && peer_error!=cudaErrorPeerAccessAlreadyEnabled) throw std::runtime_error(cudaGetErrorString(peer_error)); }
    if (p10) { CUDA_OK(cudaSetDevice(1)); cudaError_t peer_error=cudaDeviceEnablePeerAccess(0,0); if(peer_error!=cudaSuccess && peer_error!=cudaErrorPeerAccessAlreadyEnabled) throw std::runtime_error(cudaGetErrorString(peer_error)); }
    std::ofstream out(opt.output); if(!out) throw std::runtime_error("cannot open output CSV");
    out << "backend,direction,src_device,dst_device,size_bytes,size_mib,iteration,latency_ms,bandwidth_GBps,p2p_available,correctness_pass,success,error\n";
    std::vector<Direction> ds={{"gpu0_to_gpu1",0,1,true,false,false},{"gpu1_to_gpu0",1,0,true,false,false},{"gpu0_to_pinned_cpu",0,-1,false,false,true},{"pinned_cpu_to_gpu0",-1,0,false,true,false},{"gpu1_to_pinned_cpu",1,-1,false,false,true},{"pinned_cpu_to_gpu1",-1,1,false,true,false}};
    for (const auto& d:ds) for (size_t s:opt.sizes) run_one(d,s,opt,d.name=="gpu0_to_gpu1"?p01:(d.name=="gpu1_to_gpu0"?p10:false),out);
    std::cerr << "P2P GPU0->GPU1=" << p01 << " GPU1->GPU0=" << p10 << '\n'; return 0;
  } catch (const std::exception& e) { std::cerr << "fatal: " << e.what() << '\n'; return 2; }
}
