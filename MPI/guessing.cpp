#include "PCFG.h"
using namespace std;

void SendPT(const PT& pt, int dest) {
    // dest -> 目标进程编号；tag -> 消息标记，收发一致
    // content -> segment 结构
    int n = pt.content.size();
    MPI_Send(&n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Send(pt.max_indices.data(), n, MPI_INT, dest, 1, MPI_COMM_WORLD);
    MPI_Send(pt.curr_indices.data(), n, MPI_INT, dest, 2, MPI_COMM_WORLD);
    std::vector<int> types(n), lens(n);
    for (int i = 0; i < n; ++i) {
        // segment 类型
        types[i] = pt.content[i].type;
        // segment 长度
        lens[i] = pt.content[i].length;
    }
    MPI_Send(types.data(), n, MPI_INT, dest, 3, MPI_COMM_WORLD);
    MPI_Send(lens.data(), n, MPI_INT, dest, 4, MPI_COMM_WORLD);
}

void ReceivePT(PT& pt, int src) {
    int n;
    MPI_Recv(&n, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pt.max_indices.resize(n);
    pt.curr_indices.resize(n);
    pt.content.resize(n);
    MPI_Recv(pt.max_indices.data(), n, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(pt.curr_indices.data(), n, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::vector<int> types(n), lens(n);
    MPI_Recv(types.data(), n, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(lens.data(), n, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < n; ++i) {
        pt.content[i].type = types[i];
        pt.content[i].length = lens[i];
    }
}

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

void PriorityQueue::PopNext() {

    // 获取 MPI 进程信息
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 主进程分发任务：每进程分配一个 PT
    int task_num = 0;
    vector<PT> batch_pts;
    if (world_rank == 0) {
        task_num = std::min(world_size, int(priority.size()));
        for (int i = 0; i < task_num; ++i) {
            batch_pts.push_back(priority.front());
            priority.erase(priority.begin());
        }
    }
    // 广播本轮任务数
    MPI_Bcast(&task_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 主进程发送PT；其他进程接收PT
    bool has_pt = false;
    PT my_pt;
    if (world_rank == 0) {
        if (task_num > 0) {
            my_pt = batch_pts[0];
            has_pt = true;
            for (int p = 1; p < task_num; ++p) {
                SendPT(batch_pts[p], p); // 序列化 PT 发送
            }
        }
    } else {
        if (world_rank < task_num) {
            ReceivePT(my_pt, 0);
            has_pt = true;
        }
    }

    // 各进程本地串行生成
    vector<std::string> local_guesses;
    if (has_pt) {
        GenerateNew(my_pt, local_guesses);
    }
    int local_count = local_guesses.size();

    // 主进程统计结果

    // 收集猜测结果数目
    vector<int> all_counts(world_size, 0);
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 收集所有口令长度
    vector<int> local_lengths(local_guesses.size());
    for (size_t i = 0; i < local_guesses.size(); ++i)
        local_lengths[i] = local_guesses[i].size();
    vector<int> recv_counts(world_size, 0), displs(world_size, 0), all_lengths;
    // recv_counts: 每个进程将会发多少条口令长度（即每个进程有多少个口令）
    // displs: 每个进程收集到的长度数据在 root 进程 all_lengths 中的起始下标。
    // all_lengths: 最终root进程收集所有长度的数组

    // total_count: 所有口令数目
    int total_count = 0;
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            recv_counts[i] = all_counts[i];
            if (i > 0) 
                displs[i] = displs[i-1] + recv_counts[i-1];
            total_count += all_counts[i];
        }
        all_lengths.resize(total_count);
    }

    MPI_Gatherv(local_lengths.data(), local_count, MPI_INT,
                all_lengths.data(), recv_counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

                
    // 收集每个进程的字符数
    int local_chars = 0;
    for (auto& s : local_guesses) local_chars += s.size();
    vector<int> chars_counts(world_size, 0), chars_displs(world_size, 0);
    MPI_Gather(&local_chars, 1, MPI_INT, chars_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_chars = 0;
    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            if (i > 0) 
                chars_displs[i] = chars_displs[i-1] + chars_counts[i-1];
            total_chars += chars_counts[i];
        }
    }


    // 收集所有口令内容
    vector<char> local_data;
    for (auto& s : local_guesses) 
        local_data.insert(local_data.end(), s.begin(), s.end());
    vector<char> all_data;
    if (world_rank == 0) 
        all_data.resize(total_chars);

    MPI_Gatherv(local_data.data(), local_chars, MPI_CHAR,
                all_data.data(), chars_counts.data(), chars_displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);


    // 主进程合并到全局 guesses
    int offset = 0;
    if (world_rank == 0){
        for (int i = 0; i < total_count; ++i) {
            guesses.emplace_back(&all_data[offset], &all_data[offset] + all_lengths[i]);
            offset += all_lengths[i];
        }
    }
    
    last_generated_count = guesses.size();

    // 主进程更新优先队列（将出队 PT 新生成的 PT 入队）
    if (world_rank == 0 && task_num > 0) {
        for (int i = 0; i < task_num; ++i) {
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

    // 等待所有进程完成
    MPI_Barrier(MPI_COMM_WORLD);
}

void PriorityQueue::GenerateNew(const PT& pt, std::vector<std::string>& guesses) {

    if (pt.content.size() == 1) {
        segment *a = nullptr;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        for (int i = 0; i < pt.max_indices[0]; ++i) {
            guesses.push_back(a->ordered_values[i]);
        }
        
    }
    else{
        std::string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2) guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3) guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }
        segment *a = nullptr;
        if (pt.content.back().type == 1) a = &m.letters[m.FindLetter(pt.content.back())];
        if (pt.content.back().type == 2) a = &m.digits[m.FindDigit(pt.content.back())];
        if (pt.content.back().type == 3) a = &m.symbols[m.FindSymbol(pt.content.back())];
        for (int i = 0; i < pt.max_indices.back(); ++i) {
            guesses.push_back(guess + a->ordered_values[i]);
    }
    }
}


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


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
// void PriorityQueue::Generate(PT pt)
// {
//     /*
//     * 基于当前 PT 生成一系列候选口令: 使用所有可能的 value 并行填充最后一个 segment
//     */
//     // 获取MPI进程信息
//     int world_rank, world_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     // 计算PT的概率，这里主要是给PT的概率进行初始化
//     if (world_rank == 0)
//         CalProb(pt);

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

//         // // ========================= MPI 部分 =========================
//         // int total = pt.max_indices[pt.content.size() - 1];      // 最后一个segment的所有可能value数目
//         // int start = (total * world_rank) / world_size;          // 当前进程的任务起始下标
//         // int end   = (total * (world_rank + 1)) / world_size;    // 当前进程的任务结束下标

//         // // 每个进程生成一部分猜测存储到局部结果 local_guesses
//         // vector<string> local_guesses;
//         // for (int i = start; i < end; ++i)
//         // {
//         //     string temp = guess + a->ordered_values[i];
//         //     local_guesses.push_back(temp);
//         // }

//         // int local_count = local_guesses.size();                 // 当前进程生成的猜测数目
//         // int global_count = 0;                                   // 全局猜测数目
//         // MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);        // 汇总所有进程的猜测数目到主进程 rank0

//         // // =================== 变长字符串全局收集 ===================
//         // vector<int> local_lengths;
//         // vector<char> local_data;
//         // for (const string& s : local_guesses) {
//         //     local_lengths.push_back(s.size());
//         //     local_data.insert(local_data.end(), s.begin(), s.end());
//         // }
//         // int local_chars = local_data.size();

//         // // 收集每个进程的猜测生成数
//         // vector<int> all_counts(world_size);         // 主进程汇总每个进程生成的口令条数
//         // MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

//         // // 收集每个进程所有猜测的长度信息
//         // vector<int> recv_counts(world_size, 0), len_displs(world_size, 0);
//         // int total_guess_count = 0;
//         // if (world_rank == 0) {
//         //     for (int i = 0; i < world_size; ++i) {
//         //         recv_counts[i] = all_counts[i];
//         //         if (i > 0) len_displs[i] = len_displs[i-1] + recv_counts[i-1];
//         //         total_guess_count += all_counts[i];
//         //     }
//         // }
//         // vector<int> all_lengths;
//         // if (world_rank == 0) all_lengths.resize(total_guess_count);

//         // MPI_Gatherv(local_lengths.data(), local_count, MPI_INT,
//         //             all_lengths.data(), recv_counts.data(), len_displs.data(), MPI_INT,
//         //             0, MPI_COMM_WORLD);

//         // // 收集全部猜测字符串内容
//         // int local_chars_count = local_chars;
//         // std::vector<int> chars_counts(world_size, 0), chars_displs(world_size, 0);
//         // MPI_Gather(&local_chars_count, 1, MPI_INT, chars_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
//         // int total_chars = 0;
//         // if (world_rank == 0) {
//         //     for (int i = 0; i < world_size; ++i) {
//         //         if (i > 0) chars_displs[i] = chars_displs[i-1] + chars_counts[i-1];
//         //         total_chars += chars_counts[i];
//         //     }
//         // }
//         // std::vector<char> all_data;
//         // if (world_rank == 0) all_data.resize(total_chars);

//         // MPI_Gatherv(local_data.data(), local_chars, MPI_CHAR,
//         //             all_data.data(), chars_counts.data(), chars_displs.data(), MPI_CHAR,
//         //             0, MPI_COMM_WORLD);

//         // // 主进程拆分为vector<string>
//         // if (world_rank == 0) {
//         //     guesses.clear();
//         //     guesses.reserve(total_guess_count);
//         //     int offset = 0;
//         //     for (int i = 0; i < total_guess_count; ++i) {
//         //         guesses.emplace_back(&all_data[offset], &all_data[offset] + all_lengths[i]);
//         //         offset += all_lengths[i];
//         //     }
//         // }
//         // // =================== 变长字符串全局收集 END ===================
//         // ========================= MPI 部分 =========================
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

//         // ========================= MPI 部分 =========================
//         // int total = pt.max_indices[pt.content.size() - 1];      // 最后一个segment的所有可能value数目
//         // int start = (total * world_rank) / world_size;          // 当前进程的任务起始下标
//         // int end   = (total * (world_rank + 1)) / world_size;    // 当前进程的任务结束下标

//         // // 每个进程生成一部分猜测存储到局部结果 local_guesses
//         // vector<string> local_guesses;
//         // for (int i = start; i < end; ++i)
//         // {
//         //     string temp = guess + a->ordered_values[i];
//         //     local_guesses.push_back(temp);
//         // }

//         // int local_count = local_guesses.size();                 // 当前进程生成的猜测数目
//         // int global_count = 0;                                   // 全局猜测数目
//         // MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);        // 汇总所有进程的猜测数目到主进程 rank0

//         // // =================== 变长字符串全局收集 ===================
//         // vector<int> local_lengths;
//         // vector<char> local_data;
//         // for (const string& s : local_guesses) {
//         //     local_lengths.push_back(s.size());
//         //     local_data.insert(local_data.end(), s.begin(), s.end());
//         // }
//         // int local_chars = local_data.size();

//         // // 收集每个进程的猜测生成数
//         // vector<int> all_counts(world_size);         // 主进程汇总每个进程生成的口令条数
//         // MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

//         // // 收集每个进程所有猜测的长度信息
//         // // 先算总长度
//         // vector<int> recv_counts(world_size, 0), len_displs(world_size, 0);
//         // int total_guess_count = 0;
//         // if (world_rank == 0) {
//         //     for (int i = 0; i < world_size; ++i) {
//         //         recv_counts[i] = all_counts[i];
//         //         if (i > 0) len_displs[i] = len_displs[i-1] + recv_counts[i-1];
//         //         total_guess_count += all_counts[i];
//         //     }
//         // }
//         // vector<int> all_lengths;
//         // if (world_rank == 0) all_lengths.resize(total_guess_count);

//         // MPI_Gatherv(local_lengths.data(), local_count, MPI_INT,
//         //             all_lengths.data(), recv_counts.data(), len_displs.data(), MPI_INT,
//         //             0, MPI_COMM_WORLD);

//         // // 收集全部猜测字符串内容
//         // int local_chars_count = local_chars;
//         // std::vector<int> chars_counts(world_size, 0), chars_displs(world_size, 0);
//         // MPI_Gather(&local_chars_count, 1, MPI_INT, chars_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
//         // int total_chars = 0;
//         // if (world_rank == 0) {
//         //     for (int i = 0; i < world_size; ++i) {
//         //         if (i > 0) chars_displs[i] = chars_displs[i-1] + chars_counts[i-1];
//         //         total_chars += chars_counts[i];
//         //     }
//         // }
//         // std::vector<char> all_data;
//         // if (world_rank == 0) all_data.resize(total_chars);

//         // MPI_Gatherv(local_data.data(), local_chars, MPI_CHAR,
//         //             all_data.data(), chars_counts.data(), chars_displs.data(), MPI_CHAR,
//         //             0, MPI_COMM_WORLD);

//         // // 主进程拆分为vector<string>
//         // if (world_rank == 0) {
//         //     guesses.clear();
//         //     guesses.reserve(total_guess_count);
//         //     int offset = 0;
//         //     for (int i = 0; i < total_guess_count; ++i) {
//         //         guesses.emplace_back(&all_data[offset], &all_data[offset] + all_lengths[i]);
//         //         offset += all_lengths[i];
//         //     }
//         // }
//         // // =================== 变长字符串全局收集 END ===================
//         // ========================= MPI 部分 =========================
//     }
//     last_generated_count = guesses.size() - total_guesses_before;
// }
