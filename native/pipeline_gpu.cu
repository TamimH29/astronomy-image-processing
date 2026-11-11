// Resident image operations. NVRTC compiles this as CUDA C++20 for the device
// in the current worker. No host callbacks or device-specific warp assumptions.
// NVRTC supplies device math intrinsics; it need not find host C headers.
constexpr int INT_MAX = 2147483647;
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))

extern "C" __global__ void fill_mesh(double* mesh, const bool* measured,
    const int* order, const double* cx, const double* cy, int nx, int ny,
    int accepted, int box, int* error) {
    int p = blockIdx.x * blockDim.x + threadIdx.x, n = nx * ny;
    if (p >= n || measured[p]) return;
    int axes[2], dim = 1;
    if (nx > 1) axes[dim++ - 1] = 0;
    if (ny > 1) axes[dim++ - 1] = 1;
    int neighbors = min(8, accepted);
    for (;;) {
        double a[3][3] = {}, b[3] = {};
        for (int k = 0; k < neighbors; ++k) {
            int q = order[p * n + k];
            double dx = (cx[q % nx] - cx[p % nx]) / box;
            double dy = (cy[q / nx] - cy[p / nx]) / box;
            double d = dx * dx + dy * dy, v[3] = {1., 0., 0.};
            for (int j = 1; j < dim; ++j) v[j] = axes[j-1] == 0 ? dx : dy;
            double w = 1. / fmax(d, 1e-12);
            for (int r = 0; r < dim; ++r) {
                b[r] += w * v[r] * mesh[q];
                for (int c = 0; c < dim; ++c) a[r][c] += w * v[r] * v[c];
            }
        }
        double scale = 0.;
        for (int r = 0; r < dim; ++r)
            for (int c = 0; c < dim; ++c) scale = fmax(scale, fabs(a[r][c]));
        bool ok = true;
        for (int c = 0; c < dim; ++c) {
            int pivot = c;
            for (int r = c+1; r < dim; ++r)
                if (fabs(a[r][c]) > fabs(a[pivot][c])) pivot = r;
            if (fabs(a[pivot][c]) <= 1e-12 * scale) { ok = false; break; }
            for (int j = 0; j < dim; ++j) {
                double t = a[c][j]; a[c][j] = a[pivot][j]; a[pivot][j] = t;
            }
            double t = b[c]; b[c] = b[pivot]; b[pivot] = t;
            double divisor = a[c][c];
            for (int j = 0; j < dim; ++j) a[c][j] /= divisor;
            b[c] /= divisor;
            for (int r = 0; r < dim; ++r) if (r != c) {
                double factor = a[r][c];
                for (int j = 0; j < dim; ++j) a[r][j] -= factor * a[c][j];
                b[r] -= factor * b[c];
            }
        }
        if (ok) { mesh[p] = b[0]; return; }
        if (neighbors == accepted) { atomicExch(error, 1); return; }
        neighbors = min(2 * neighbors, accepted);
    }
}

// Eight local Jacobi steps per active 32x8 tile. Global iterations revisit
// changed tiles and their neighbors. Every update is monotone; convergence
// gives the same result regardless of block execution order.
extern "C" __global__ void reconstruct(const double* input, double* output,
    const double* mask, const int* labels, int h, int w, int mode,
    const int* active, int* next_active) {
    int bx = blockIdx.x, by = blockIdx.y, tilesx = gridDim.x;
    if (!active[by * tilesx + bx]) return;
    constexpr int stride = 34, shared_n = 340;
    __shared__ double a[shared_n], b[shared_n];
    __shared__ int lab[shared_n];
    __shared__ int changed;
    int t = threadIdx.y * 32 + threadIdx.x;
    for (int k = t; k < shared_n; k += 256) {
        int y = by*8 + k/stride - 1, x = bx*32 + k%stride - 1;
        bool inside = y >= 0 && y < h && x >= 0 && x < w;
        lab[k] = inside ? labels[y*w+x] : 0;
        a[k] = b[k] = inside ? input[y*w+x] : 0.;
    }
    if (!t) changed = 0;
    __syncthreads();
    int y = by*8 + threadIdx.y, x = bx*32 + threadIdx.x;
    int k = (threadIdx.y+1)*stride + threadIdx.x+1;
    bool inside = y < h && x < w;
    double original = a[k];
    for (int step = 0; step < 8; ++step) {
        double value = a[k];
        if (inside && lab[k]) {
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) if (dy || dx) {
                    int q = k + dy*stride + dx;
                    if (lab[q] != lab[k]) continue;
                    if (mode == 0) value = fmax(value, a[q]);
                    else if (mode == 1) value = fmin(value, a[q]);
                    else {
                        int gy = y+dy, gx = x+dx;
                        if (mask[gy*w+gx] == mask[y*w+x]) value = fmin(value, a[q]+1.);
                    }
                }
            if (mode == 0) value = fmin(mask[y*w+x], value);
            if (mode == 1) value = fmax(mask[y*w+x], value);
        }
        b[k] = value;
        __syncthreads();
        for (int j = t; j < shared_n; j += 256) a[j] = b[j];
        __syncthreads();
    }
    if (inside) {
        output[y*w+x] = a[k];
        if (a[k] != original) atomicExch(&changed, 1);
    }
    __syncthreads();
    if (!t && changed) {
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                int yy = by+dy, xx = bx+dx;
                if (yy >= 0 && yy < gridDim.y && xx >= 0 && xx < tilesx)
                    atomicExch(next_active + yy*tilesx+xx, 1);
            }
    }
}

extern "C" __global__ void choose_peaks(const long long* pixels, const int* parent,
    const int* starts, const int* state, int* winner, int n, int w, int separation) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    winner[i] = 0;
    if (state[i]) return;
    long long p = pixels[i];
    for (int j = starts[parent[i]]; j < i; ++j) if (!state[j]) {
        long long q = pixels[j], dy = p/w-q/w, dx = p%w-q%w;
        if (dy*dy+dx*dx < separation*separation) return;
    }
    winner[i] = 1;
}
extern "C" __global__ void suppress_peaks(const long long* pixels, const int* parent,
    const int* starts, int* state, const int* winner, int n, int w, int separation) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n || state[i]) return;
    if (winner[i]) { state[i] = 1; return; }
    long long p = pixels[i];
    for (int j = starts[parent[i]]; j < i; ++j) if (winner[j]) {
        long long q = pixels[j], dy = p/w-q/w, dx = p%w-q%w;
        if (dy*dy+dx*dx < separation*separation) { state[i] = -1; return; }
    }
}

extern "C" __global__ void descent(const double* height, const int* labels,
    const bool* markers, int* successor, double* distance, int h, int w) {
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    if (p >= h*w) return;
    successor[p] = p;
    distance[p] = 0.;
    if (!labels[p] || markers[p]) return;
    int y = p/w, x = p%w, best = p;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) if (dy || dx) {
            int yy=y+dy, xx=x+dx, q=yy*w+xx;
            if (yy < 0 || yy >= h || xx < 0 || xx >= w || labels[q] != labels[p]) continue;
            if (height[q] < height[best] || (height[q] == height[best] && height[q] < height[p] && q < best)) best = q;
        }
    successor[p] = best;
    if (best == p) distance[p] = INFINITY;
}
extern "C" __global__ void plateau_paths(const double* height, const double* distance,
    const int* labels, int* successor, int h, int w) {
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    if (p >= h*w || !labels[p] || distance[p] == 0.) return;
    int y=p/w, x=p%w, best=INT_MAX;
    for (int dy=-1; dy<=1; ++dy)
        for (int dx=-1; dx<=1; ++dx) if (dy || dx) {
            int yy=y+dy, xx=x+dx, q=yy*w+xx;
            if (yy>=0 && yy<h && xx>=0 && xx<w && labels[q]==labels[p]
                && height[q]==height[p] && distance[q]+1.==distance[p]) best=min(best,q);
        }
    if (best != INT_MAX) successor[p]=best;
}
extern "C" __global__ void jump(const int* input, int* output, int n) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<n) output[p]=input[input[p]];
}

// Stable label/raster gathering plus a fixed tree per source avoids unordered
// floating-point atomics. Flux uses the full basin; split centroids use only
// its peak-centered window, exactly as the CPU definition does.
extern "C" __global__ void measure(const float* residual, const long long* pixels,
    const long long* starts, const long long* peaks, const bool* split,
    double* rows, int w, int radius) {
    int label=blockIdx.x, t=threadIdx.x;
    long long start=starts[label], end=starts[label+1], peak=peaks[label];
    int px=peak%w, py=peak/w;
    double v[8]={};
    int xmin=INT_MAX, ymin=INT_MAX, xmax=-1, ymax=-1;
    for (long long i=start+t; i<end; i+=256) {
        long long p=pixels[i]; int x=p%w, y=p/w;
        xmin=min(xmin,x); ymin=min(ymin,y); xmax=max(xmax,x); ymax=max(ymax,y);
        double raw=residual[p]; v[0]+=raw;
        if (!split[label] || (abs(x-px)<=radius && abs(y-py)<=radius)) {
            double weight=fmax(raw,0.);
            v[1]+=weight; v[2]+=weight*x; v[3]+=weight*y;
            v[4]+=x; v[5]+=y; v[6]+=1.;
        }
    }
    __shared__ double sums[7][256];
    __shared__ int boxes[4][256];
    for (int j=0;j<7;++j) sums[j][t]=v[j];
    boxes[0][t]=xmin; boxes[1][t]=ymin; boxes[2][t]=xmax; boxes[3][t]=ymax;
    __syncthreads();
    for (int s=128;s;s/=2) {
        if(t<s) {
            for(int j=0;j<7;++j) sums[j][t]+=sums[j][t+s];
            boxes[0][t]=min(boxes[0][t],boxes[0][t+s]);
            boxes[1][t]=min(boxes[1][t],boxes[1][t+s]);
            boxes[2][t]=max(boxes[2][t],boxes[2][t+s]);
            boxes[3][t]=max(boxes[3][t],boxes[3][t+s]);
        }
        __syncthreads();
    }
    if(!t) {
        rows[label*8]=sums[1][0]>0 ? sums[2][0]/sums[1][0] : sums[4][0]/sums[6][0];
        rows[label*8+1]=sums[1][0]>0 ? sums[3][0]/sums[1][0] : sums[5][0]/sums[6][0];
        for(int j=0;j<4;++j) rows[label*8+2+j]=boxes[j][0];
        rows[label*8+6]=end-start; rows[label*8+7]=sums[0][0];
    }
}
