#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
using namespace std;

// ========================== GPU Kernel =============================
__global__ void generate_guesses_kernel(
    const char* prefix_data,
    const int* prefix_offsets,
    const int* prefix_lens,
    const char* suffix_data,
    const int* suffix_offsets,
    const int* suffix_lens,
    const int* suffix_counts,
    char* output,
    const int* output_offsets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 定位该 tid 属于哪个 PT
    int pt_idx = 0;
    int global_offset = 0;
    while (pt_idx < gridDim.x && tid >= global_offset + suffix_counts[pt_idx]) {
        global_offset += suffix_counts[pt_idx];
        pt_idx++;
    }
    if (pt_idx >= gridDim.x) return;

    int local_tid = tid - global_offset;
    int pre_len = prefix_lens[pt_idx];
    int suf_len = suffix_lens[pt_idx];

    const char* prefix_ptr = prefix_data + prefix_offsets[pt_idx];
    const char* suffix_ptr = suffix_data + suffix_offsets[pt_idx] + local_tid * suf_len;
    char* out_ptr = output + output_offsets[tid];

    for (int i = 0; i < pre_len; ++i)
        out_ptr[i] = prefix_ptr[i];
    for (int i = 0; i < suf_len; ++i)
        out_ptr[pre_len + i] = suffix_ptr[i];

    out_ptr[pre_len + suf_len] = '\0';
}
// ========================== GPU Kernel =============================

void PriorityQueue::CalProb(PT &pt)
{
    /*
    * 对一个PT，计算其除最后一个 segment 外其余 segment 均被实例化下的一个联合概率
    */
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    /*
    * 对模型统计到的所有 preterminal 初始化，按概率构造优先队列
    * 初始化跳过最后一个 segment，由后续并行生成猜测时填充
    */
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                //* m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                //* m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                //* m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext() {
    const int batch_size = min(8, int(priority.size()));

    if (batch_size == 0) return;

    vector<PT> batch_pts;
    for (int i = 0; i < batch_size; ++i) {
        batch_pts.push_back(priority.front());
        priority.erase(priority.begin());
    }

    // CUDA 多 PT 并行生成所有猜测
    GenerateBatch(batch_pts); 

    // 所有出队 PT 的 NewPTs 入队
    for (int i = 0; i < batch_size; ++i) {
        vector<PT> new_pts = batch_pts[i].NewPTs();
        for (PT pt : new_pts) {
            CalProb(pt);
            for (auto iter = priority.begin(); iter != priority.end(); iter++)
            {
                if (iter != priority.end() - 1 && iter != priority.begin())
                {
                    if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                    {
                        priority.emplace(iter + 1, pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1)
                {
                    priority.emplace_back(pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < pt.prob)
                {
                    priority.emplace(iter, pt);
                    break;
                }
            }
        }
    }
}


// void PriorityQueue::PopNext()
// {
//     /*
//     * 在优先队列中取出头部 PT 作为猜测
//     * 根据这个出队 PT，改造生成若干新的 PT
//     */

//     // ========================= 查看猜测分布 =========================
//     last_pt = priority.front();
//     // ========================= 查看猜测分布 =========================

//     // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
//     Generate(priority.front());

//     // 然后需要根据即将出队的PT，生成一系列新的PT
//     vector<PT> new_pts = priority.front().NewPTs();
//     for (PT pt : new_pts)
//     {
//         // 计算概率
//         CalProb(pt);
//         // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
//         for (auto iter = priority.begin(); iter != priority.end(); iter++)
//         {
//             // 对于非队首和队尾的特殊情况
//             if (iter != priority.end() - 1 && iter != priority.begin())
//             {
//                 // 判定概率
//                 if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
//                 {
//                     priority.emplace(iter + 1, pt);
//                     break;
//                 }
//             }
//             if (iter == priority.end() - 1)
//             {
//                 priority.emplace_back(pt);
//                 break;
//             }
//             if (iter == priority.begin() && iter->prob < pt.prob)
//             {
//                 priority.emplace(iter, pt);
//                 break;
//             }
//         }
//     }

//     // 现在队首的PT善后工作已经结束，将其出队（删除）
//     priority.erase(priority.begin());
// }

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    /*
    * 根据当前PT生成一系列新的PT的方法函数
    */
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

void GenerateBatch_CPU(const vector<PT>& cpu_pts) {
    vector<thread> threads;
    for (const auto& pt : cpu_pts) {
        threads.emplace_back([&, pt]() {
            vector<string> local;
            GenerateNew(pt, local); 
            #pragma omp critical
            {
                guesses.insert(guesses.end(), local.begin(), local.end());
                total_guesses += local.size();
            }
        });
    }
    for (auto& t : threads) t.join();
}

void PriorityQueue::GenerateBatch(const vector<PT>& pts) {
    
    vector<string> prefixes;                    // 记录每个 PT 的前缀串
    vector<vector<string>> suffix_groups;       // 记录每个 PT 的所有尾部填充 value
    vector<int> prefix_lens;                    // 记录每个 PT 的前缀串的长度
    vector<int> suffix_lens;                    // 记录每个 PT 的填充 value 的长度（每个 PT 的所有候选 value 等长）
    vector<int> suffix_counts;                  // 记录每个 PT 的填充 value 的数目

    for (const auto& pt : pts) {
        string prefix = "";
        int idx = 0;
        for (idx = 0; idx < pt.content.size() - 1; ++idx) {
            int type = pt.content[idx].type;
            int val = pt.curr_indices[idx];
            if (type == 1) prefix += m.letters[m.FindLetter(pt.content[idx])].ordered_values[val];
            if (type == 2) prefix += m.digits[m.FindDigit(pt.content[idx])].ordered_values[val];
            if (type == 3) prefix += m.symbols[m.FindSymbol(pt.content[idx])].ordered_values[val];
        }
        prefixes.push_back(prefix);
        prefix_lens.push_back(prefix.length());

        segment* a = nullptr;
        if (pt.content[pt.content.size() - 1].type == 1) a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2) a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3) a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        suffix_groups.push_back(a->ordered_values);
        suffix_counts.push_back(a->ordered_values.size());
        suffix_lens.push_back(a->ordered_values[0].size());
    }

    //  prefix 和 suffix 转 char 数组 与 偏移量
    vector<char> h_prefix_data;
    vector<int> h_prefix_offsets;
    for (const auto& s : prefixes) {
        h_prefix_offsets.push_back(h_prefix_data.size());
        h_prefix_data.insert(h_prefix_data.end(), s.begin(), s.end());
    }

    vector<char> h_suffix_data;
    vector<int> h_suffix_offsets;
    for (const auto& group : suffix_groups) {
        h_suffix_offsets.push_back(h_suffix_data.size());
        for (const auto& val : group) {
            h_suffix_data.insert(h_suffix_data.end(), val.begin(), val.end());
        }
    }

    // 输出起始偏移
    int total_count = 0;        // 多 PT 生成的所有猜测口令
    for (int c : suffix_counts) total_count += c;

    vector<int> h_output_offsets(total_count);
    vector<int> guess_lens(total_count);
    int acc = 0;    // 累加器
    int k = 0;      // 猜测口令游标
    for (int i = 0; i < pts.size(); ++i) {
        int total_len = prefix_lens[i] + suffix_lens[i] + 1;    // 每个 PT 猜测口令的长度
        for (int j = 0; j < suffix_counts[i]; ++j) {
            h_output_offsets[k] = acc;
            guess_lens[k] = total_len;
            acc += total_len;
            ++k;
        }
    }

    // 分配 device 显存
    char *d_prefix, *d_suffix, *d_output;
    int *d_prefix_offsets, *d_prefix_lens;
    int *d_suffix_offsets, *d_suffix_lens, *d_suffix_counts;
    int *d_output_offsets;

    cudaMalloc(&d_prefix, h_prefix_data.size());
    cudaMalloc(&d_prefix_offsets, sizeof(int) * h_prefix_offsets.size());
    cudaMalloc(&d_prefix_lens, sizeof(int) * prefix_lens.size());

    cudaMalloc(&d_suffix, h_suffix_data.size());
    cudaMalloc(&d_suffix_offsets, sizeof(int) * h_suffix_offsets.size());
    cudaMalloc(&d_suffix_lens, sizeof(int) * suffix_lens.size());
    cudaMalloc(&d_suffix_counts, sizeof(int) * suffix_counts.size());

    cudaMalloc(&d_output_offsets, sizeof(int) * h_output_offsets.size());
    cudaMalloc(&d_output, acc);

    // 拷贝 host -> device
    cudaMemcpy(d_prefix, h_prefix_data.data(), h_prefix_data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_offsets, h_prefix_offsets.data(), sizeof(int) * h_prefix_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_lens, prefix_lens.data(), sizeof(int) * prefix_lens.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(d_suffix, h_suffix_data.data(), h_suffix_data.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_offsets, h_suffix_offsets.data(), sizeof(int) * h_suffix_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_lens, suffix_lens.data(), sizeof(int) * suffix_lens.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_suffix_counts, suffix_counts.data(), sizeof(int) * suffix_counts.size(), cudaMemcpyHostToDevice);

    cudaMemcpy(d_output_offsets, h_output_offsets.data(), sizeof(int) * h_output_offsets.size(), cudaMemcpyHostToDevice);

    // 启动 kernel
    int BLOCK_SIZE = 256;
    int GRID_SIZE = (total_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid(GRID_SIZE, 1);
    generate_guesses_kernel<<<dimGrid, dimBlock>>>(
        d_prefix, d_prefix_offsets, d_prefix_lens,
        d_suffix, d_suffix_offsets, d_suffix_lens, d_suffix_counts,
        d_output, d_output_offsets);

    // 拷贝 device -> host
    vector<char> h_output(acc);
    cudaMemcpy(h_output.data(), d_output, acc, cudaMemcpyDeviceToHost);

    // 写入结果
    for (int i = 0; i < total_count; ++i)
        guesses.emplace_back(string(&h_output[h_output_offsets[i]]));

    total_guesses += total_count;
    last_generated_count = total_count;

    // 清理内存
    cudaFree(d_prefix);
    cudaFree(d_prefix_offsets);
    cudaFree(d_prefix_lens);
    cudaFree(d_suffix);
    cudaFree(d_suffix_offsets);
    cudaFree(d_suffix_lens);
    cudaFree(d_suffix_counts);
    cudaFree(d_output_offsets);
    cudaFree(d_output);
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// void PriorityQueue::Generate(PT pt)
// {
//     /*
//     * 基于当前 PT 生成一系列候选口令: 使用所有可能的 value 并行填充最后一个 segment
//     */
//     // 计算PT的概率，这里主要是给PT的概率进行初始化
//     CalProb(pt);

//     int total_guesses_before = guesses.size();

//     // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
//     if (pt.content.size() == 1)
//     {
//         // PT 猜测的 segment 前缀，由于这时候只有一个segment，所以前缀置空
//         string guess = "";

//         // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
//         segment *a;
//         // 在模型中定位到这个segment
//         if (pt.content[0].type == 1)
//         {
//             a = &m.letters[m.FindLetter(pt.content[0])];
//         }
//         if (pt.content[0].type == 2)
//         {
//             a = &m.digits[m.FindDigit(pt.content[0])];
//         }
//         if (pt.content[0].type == 3)
//         {
//             a = &m.symbols[m.FindSymbol(pt.content[0])];
//         }
        
//         // TODO (Multi-thread)：
//         // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
//         // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
//         // 这个过程是可以高度并行化的
//         // ========================= 串行部分 =========================
//         // for (int i = 0; i < pt.max_indices[0]; i += 1)
//         // {
//         //     string guess = a->ordered_values[i];
//         //     // cout << guess << endl;
//         //     guesses.emplace_back(guess);
//         //     total_guesses += 1;
//         // }
//         // ========================= 串行部分 =========================

//         // ========================= GPU 加速 =========================
//         // CUDA 加速区域
//         int count = a->ordered_values.size();               // 填充 value 数目
//         int prefix_len = guess.length();                    // 前缀长度
//         int suffix_len = a->ordered_values[0].length();     // value 长度
//         int total_len = prefix_len + suffix_len + 1;        // 总长度

//         // value 转 char 长数组
//         char* h_suffixes = new char[count * suffix_len];
//         for (int i = 0; i < count; ++i)
//             memcpy(h_suffixes + i * suffix_len, a->ordered_values[i].c_str(), suffix_len);

//         // 分配 device 显存
//         char* d_prefix;             // 保存前缀
//         char* d_suffixes;           // 保存 value
//         char* d_output;             // 保存输出

//         cudaMalloc(&d_prefix, prefix_len);
//         cudaMalloc(&d_suffixes, count * suffix_len);
//         cudaMalloc(&d_output, count * total_len);

//         // host 拷贝数据到 device
//         cudaMemcpy(d_prefix, guess.c_str(), prefix_len, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_suffixes, h_suffixes, count * suffix_len, cudaMemcpyHostToDevice);

//         // 启动 kernel
//         int BLOCK_SIZE = 256;
//         int GRID_SIZE = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

//         dim3 dimBlock(BLOCK_SIZE, 1);
//         dim3 dimGrid(GRID_SIZE, 1);

//         generate_guesses_kernel<<<dimGrid, dimBlock>>>(
//             d_prefix, prefix_len,
//             d_suffixes, suffix_len,
//             d_output, total_len, count
//         );

//         // device 拷贝结果到 host
//         char* h_output = new char[count * total_len];
//         cudaMemcpy(h_output, d_output, count * total_len, cudaMemcpyDeviceToHost);

//         // 转成 string 填充 guesses
//         for (int i = 0; i < count; ++i)
//             guesses.emplace_back(string(h_output + i * total_len));

//         total_guesses += count;

//         // 清理内存
//         delete[] h_suffixes;
//         delete[] h_output;
//         cudaFree(d_prefix);
//         cudaFree(d_suffixes);
//         cudaFree(d_output);
//         // ========================= GPU 加速 =========================
//     }
//     else
//     {
//         string guess;
//         int seg_idx = 0;
//         // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
//         // segment值根据curr_indices中对应的值加以确定
//         // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
//         for (int idx : pt.curr_indices)
//         {
//             if (pt.content[seg_idx].type == 1)
//             {
//                 guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 2)
//             {
//                 guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             }
//             if (pt.content[seg_idx].type == 3)
//             {
//                 guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//             }
//             seg_idx += 1;
//             if (seg_idx == pt.content.size() - 1)
//             {
//                 break;
//             }
//         }

//         // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
//         segment *a;
//         if (pt.content[pt.content.size() - 1].type == 1)
//         {
//             a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
//         }
//         if (pt.content[pt.content.size() - 1].type == 2)
//         {
//             a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
//         }
//         if (pt.content[pt.content.size() - 1].type == 3)
//         {
//             a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
//         }
        
//         // TODO (Multi-thread)：
//         // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
//         // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
//         // 这个过程是可以高度并行化的
//         // ========================= 串行部分 =========================
//         // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
//         // {
//         //     // 拼接上最后一个 segment 的实例
//         //     string temp = guess + a->ordered_values[i];
//         //     // cout << temp << endl;
//         //     guesses.emplace_back(temp);
//         //     total_guesses += 1;
//         // }
//         // ========================= 串行部分 =========================

//         // ========================= GPU 加速 =========================
//         // CUDA 加速区域
//         int count = a->ordered_values.size();               // 填充 value 数目
//         int prefix_len = guess.length();                    // 前缀长度
//         int suffix_len = a->ordered_values[0].length();     // value 长度
//         int total_len = prefix_len + suffix_len + 1;        // 总长度

//         // value 转 char 长数组
//         char* h_suffixes = new char[count * suffix_len];
//         for (int i = 0; i < count; ++i)
//             memcpy(h_suffixes + i * suffix_len, a->ordered_values[i].c_str(), suffix_len);

//         // 分配 device 显存
//         char* d_prefix;             // 保存前缀
//         char* d_suffixes;           // 保存 value
//         char* d_output;             // 保存输出

//         cudaMalloc(&d_prefix, prefix_len);
//         cudaMalloc(&d_suffixes, count * suffix_len);
//         cudaMalloc(&d_output, count * total_len);

//         // host 拷贝数据到 device
//         cudaMemcpy(d_prefix, guess.c_str(), prefix_len, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_suffixes, h_suffixes, count * suffix_len, cudaMemcpyHostToDevice);

//         // 启动 kernel
//         int BLOCK_SIZE = 256;
//         int GRID_SIZE = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

//         dim3 dimBlock(BLOCK_SIZE, 1);
//         dim3 dimGrid(GRID_SIZE, 1);

//         generate_guesses_kernel<<<dimGrid, dimBlock>>>(
//             d_prefix, prefix_len,
//             d_suffixes, suffix_len,
//             d_output, total_len, count
//         );

//         // device 拷贝结果到 host
//         char* h_output = new char[count * total_len];
//         cudaMemcpy(h_output, d_output, count * total_len, cudaMemcpyDeviceToHost);

//         // 转成 string 填充 guesses
//         for (int i = 0; i < count; ++i)
//             guesses.emplace_back(string(h_output + i * total_len));

//         total_guesses += count;

//         // 清理内存
//         delete[] h_suffixes;
//         delete[] h_output;
//         cudaFree(d_prefix);
//         cudaFree(d_suffixes);
//         cudaFree(d_output);
//         // ========================= GPU 加速 =========================
//     }
//     last_generated_count = guesses.size() - total_guesses_before;
// }