#include "PCFG.h"
using namespace std;

void* thread_worker(void* arg) 
{
    /*
    * 线程工作函数
    */
    ThreadArgs* args = (ThreadArgs*)arg;
    vector<string> local_results;

    int local_count = 0;
    for (int i = args->start; i < args->end; ++i) 
    {
        string temp = args->prefix + args->seg_ptr->ordered_values[i];

        local_results.push_back(temp);

        local_count++;
    }

    // 一次性合并到全局 guesses
    pthread_mutex_lock(args->guess_lock);
    for (const auto& g : local_results)
        args->guesses->push_back(g);
    pthread_mutex_unlock(args->guess_lock);

    // 一次性合并到全局 total_guesses
    pthread_mutex_lock(args->counter_lock);
    *(args->total_guesses) += local_count;
    pthread_mutex_unlock(args->counter_lock);

    pthread_exit(nullptr);
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

void PriorityQueue::PopNext()
{
    /*
    * 在优先队列中取出头部 PT 作为猜测
    * 根据这个出队 PT，改造生成若干新的 PT
    */

    // ========================= 查看猜测分布 =========================
    last_pt = priority.front();
    // ========================= 查看猜测分布 =========================

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
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

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
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
void PriorityQueue::Generate(PT pt)
{
    /*
    * 基于当前 PT 生成一系列候选口令: 使用所有可能的 value 并行填充最后一个 segment
    */
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    int total_guesses_before = guesses.size();

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // PT 猜测的 segment 前缀，由于这时候只有一个segment，所以前缀置空
        string guess = "";

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // TODO (Multi-thread)：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // ========================= 串行部分 =========================
        // for (int i = 0; i < pt.max_indices[0]; i += 1)
        // {
        //     string guess = a->ordered_values[i];
        //     // cout << guess << endl;
        //     guesses.emplace_back(guess);
        //     total_guesses += 1;
        // }
        // ========================= 串行部分 =========================

        // ========================= Pthread部分 =========================
        // // 数据预设
        // int value_count = pt.max_indices[pt.content.size() - 1];                // 当前 PT 最后一个 segment 的所有可能 value 数目
        // int thread_count = min(4, value_count);                                 // 线程数目
        // int chunk_size = (value_count + thread_count - 1) / thread_count;       // 每个线程分配的任务数（待填充的 value 数），向上取整

        // // 创建线程本体
        // vector<pthread_t> threads(thread_count);
        // // 创建线程参数框架
        // vector<ThreadArgs> thread_args(thread_count);

        // // 创建互斥锁
        // pthread_mutex_t guess_lock;
        // pthread_mutex_init(&guess_lock, nullptr);

        // pthread_mutex_t counter_lock;
        // pthread_mutex_init(&counter_lock, nullptr);

        // // 创建线程任务
        // for (int t = 0; t < thread_count; ++t) 
        // {
        //     int start = t * chunk_size;
        //     int end = min(start + chunk_size, value_count);

        //     thread_args[t] = 
        //     {
        //         guess,
        //         a,
        //         start,
        //         end,
        //         &guesses,
        //         &guess_lock,
        //         &total_guesses,
        //         &counter_lock
        //     };

        //     pthread_create(&threads[t], nullptr, thread_worker, &thread_args[t]);
        // }

        // // 等待所有线程完成
        // for (int t = 0; t < thread_count; ++t) 
        // {
        //     pthread_join(threads[t], nullptr);
        // }

        // // 销毁锁
        // pthread_mutex_destroy(&guess_lock);
        // pthread_mutex_destroy(&counter_lock);

        // ========================= Pthread部分 =========================

        // ========================= OpenMP部分  =========================
        // int value_count = pt.max_indices[pt.content.size() - 1];

        // // 获取线程数
        // omp_set_num_threads(4);
        // int thread_num = omp_get_max_threads();

        // // 每个线程维护局部 vector 和 count
        // vector<vector<string>> thread_guesses(thread_num);
        // vector<int> thread_counts(thread_num, 0);

        // #pragma omp parallel
        // {
        //     // 获取当前线程的编号
        //     int tid = omp_get_thread_num();
        //     // 每个线程维护一个局部 vector 和 count
        //     vector<string>& local_vec = thread_guesses[tid];
        //     int local_count = 0;

        //     // 对循环任务进行静态划分
        //     #pragma omp for schedule(static)
        //     for (int i = 0; i < value_count; ++i) 
        //     {
        //         string temp = guess + a->ordered_values[i];
        //         local_vec.push_back(temp);
        //         local_count++;
        //     }

        //     thread_counts[tid] = local_count;
        // }

        // // 合并所有线程的结果
        // for (auto& vec : thread_guesses) 
        // {
        //     guesses.insert(guesses.end(), vec.begin(), vec.end());
        // }
        // for (int cnt : thread_counts) 
        // {
        //     total_guesses += cnt;
        // }
        // ========================= OpenMP部分 =========================

        // ========================= Pthread改进部分 =========================
        // int value_count = pt.max_indices[pt.content.size() - 1];

        // if (value_count < PARALLEL_THRESHOLD) 
        // {
        //     // ======== 小任务，串行处理 ========
        //     for (int i = 0; i < value_count; ++i) 
        //     {
        //         string temp = guess + a->ordered_values[i];
        //         guesses.emplace_back(temp);
        //         total_guesses += 1;
        //     }
        // }
        // else 
        // {
        //     // ======== 大任务，Pthread并行处理 ========

        //     int thread_count = min(4, value_count);
        //     int chunk_size = (value_count + thread_count - 1) / thread_count;

        //     // 创建线程与参数
        //     vector<pthread_t> threads(thread_count);
        //     vector<ThreadArgs> thread_args(thread_count);

        //     pthread_mutex_t guess_lock;
        //     pthread_mutex_init(&guess_lock, nullptr);

        //     pthread_mutex_t counter_lock;
        //     pthread_mutex_init(&counter_lock, nullptr);

        //     for (int t = 0; t < thread_count; ++t) 
        //     {
        //         int start = t * chunk_size;
        //         int end = min(start + chunk_size, value_count);

        //         thread_args[t] = {
        //             guess,
        //             a,
        //             start,
        //             end,
        //             &guesses,
        //             &guess_lock,
        //             &total_guesses,
        //             &counter_lock
        //         };

        //         pthread_create(&threads[t], nullptr, thread_worker, &thread_args[t]);
        //     }

        //     // 等待线程结束
        //     for (int t = 0; t < thread_count; ++t) 
        //     {
        //         pthread_join(threads[t], nullptr);
        //     }

        //     pthread_mutex_destroy(&guess_lock);
        //     pthread_mutex_destroy(&counter_lock);
        // }
        // ========================= Pthread改进部分 =========================

        // ========================= OpenMP改进部分  =========================
        // 修改 Generate 函数中的并行处理部分
        int value_count = pt.max_indices[pt.content.size() - 1];

        if (value_count < PARALLEL_THREADSHOLD) 
        {
            // ================= 串行处理 =================
            for (int i = 0; i < value_count; ++i) 
            {
                string temp = guess + a->ordered_values[i];
                guesses.push_back(temp);
                total_guesses++;
            }
        } 
        else 
        {
            // ================= 并行处理 =================
            omp_set_num_threads(4);
            int thread_num = omp_get_max_threads();
            vector<vector<string>> thread_guesses(thread_num);
            vector<int> thread_counts(thread_num, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                vector<string>& local_vec = thread_guesses[tid];
                int local_count = 0;

                #pragma omp for schedule(static)
                for (int i = 0; i < value_count; ++i) 
                {
                    string temp = guess + a->ordered_values[i];
                    local_vec.push_back(temp);
                    local_count++;
                }

                thread_counts[tid] = local_count;
            }

            int total = 0;
            for (int cnt : thread_counts) 
            {
                total += cnt;
            }
            total_guesses += total;

            guesses.reserve(guesses.size() + total);
            for (auto& vec : thread_guesses) 
            {
                guesses.insert(guesses.end(), vec.begin(), vec.end());
            }
        }
        // ========================= OpenMP改进部分  =========================
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // TODO (Multi-thread)：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        // ========================= 串行部分 =========================
        // for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        // {
        //     // 拼接上最后一个 segment 的实例
        //     string temp = guess + a->ordered_values[i];
        //     // cout << temp << endl;
        //     guesses.emplace_back(temp);
        //     total_guesses += 1;
        // }
        // ========================= 串行部分 =========================

        // ========================= Pthread部分 =========================
        // // 数据预设
        // int value_count = pt.max_indices[pt.content.size() - 1];            // 当前 PT 最后一个 segment 的所有可能 value 数目
        // int thread_count = min(4, value_count);                             // 线程数目 
        // int chunk_size = (value_count + thread_count - 1) / thread_count;   // 每个线程分配的任务数（待填充的 value 数），向上取整

        // // 创建线程框架
        // vector<pthread_t> threads(thread_count);
        // vector<ThreadArgs> thread_args(thread_count);

        // pthread_mutex_t guess_lock;
        // pthread_mutex_init(&guess_lock, nullptr);

        // pthread_mutex_t counter_lock;
        // pthread_mutex_init(&counter_lock, nullptr);

        // // 创建线程任务
        // for (int t = 0; t < thread_count; ++t) 
        // {
        //     int start = t * chunk_size;
        //     int end = min(start + chunk_size, value_count);

        //     thread_args[t] = 
        //     {
        //         guess,
        //         a,
        //         start,
        //         end,
        //         &guesses,
        //         &guess_lock,
        //         &total_guesses,
        //         &counter_lock
        //     };

        //     pthread_create(&threads[t], nullptr, thread_worker, &thread_args[t]);
        // }

        // // 等待所有线程完成
        // for (int t = 0; t < thread_count; ++t) 
        // {
        //     pthread_join(threads[t], nullptr);
        // }

        // // 销毁锁
        // pthread_mutex_destroy(&guess_lock);
        // pthread_mutex_destroy(&counter_lock);

        // ========================= Pthread部分 =========================

        // ========================= OpenMP部分  =========================
        // int value_count = pt.max_indices[pt.content.size() - 1];

        // // 获取线程数
        // omp_set_num_threads(4);
        // int thread_num = omp_get_max_threads();

        // // 每个线程维护局部 vector 和 count
        // vector<vector<string>> thread_guesses(thread_num);
        // vector<int> thread_counts(thread_num, 0);

        // #pragma omp parallel
        // {
        //     int tid = omp_get_thread_num();
        //     vector<string>& local_vec = thread_guesses[tid];
        //     int local_count = 0;

        //     #pragma omp for schedule(static)
        //     for (int i = 0; i < value_count; ++i) {
        //         string temp = guess + a->ordered_values[i];
        //         local_vec.push_back(temp);
        //         local_count++;
        //     }

        //     thread_counts[tid] = local_count;
        // }

        // // 合并所有线程的结果
        // for (auto& vec : thread_guesses) 
        // {
        //     guesses.insert(guesses.end(), vec.begin(), vec.end());
        // }
        // for (int cnt : thread_counts) 
        // {
        //     total_guesses += cnt;
        // }
        // ========================= OpenMP部分 =========================

        // ========================= Pthread改进部分 =========================
        // int value_count = pt.max_indices[pt.content.size() - 1];

        // if (value_count < PARALLEL_THRESHOLD) 
        // {
        //     // ======== 小任务，串行处理 ========
        //     for (int i = 0; i < value_count; ++i) 
        //     {
        //         string temp = guess + a->ordered_values[i];
        //         guesses.emplace_back(temp);
        //         total_guesses += 1;
        //     }
        // }
        // else 
        // {
        //     // ======== 大任务，Pthread并行处理 ========

        //     int thread_count = min(4, value_count);
        //     int chunk_size = (value_count + thread_count - 1) / thread_count;

        //     // 创建线程与参数
        //     vector<pthread_t> threads(thread_count);
        //     vector<ThreadArgs> thread_args(thread_count);

        //     pthread_mutex_t guess_lock;
        //     pthread_mutex_init(&guess_lock, nullptr);

        //     pthread_mutex_t counter_lock;
        //     pthread_mutex_init(&counter_lock, nullptr);

        //     for (int t = 0; t < thread_count; ++t) 
        //     {
        //         int start = t * chunk_size;
        //         int end = min(start + chunk_size, value_count);

        //         thread_args[t] = {
        //             guess,
        //             a,
        //             start,
        //             end,
        //             &guesses,
        //             &guess_lock,
        //             &total_guesses,
        //             &counter_lock
        //         };

        //         pthread_create(&threads[t], nullptr, thread_worker, &thread_args[t]);
        //     }

        //     // 等待线程结束
        //     for (int t = 0; t < thread_count; ++t) 
        //     {
        //         pthread_join(threads[t], nullptr);
        //     }

        //     pthread_mutex_destroy(&guess_lock);
        //     pthread_mutex_destroy(&counter_lock);
        // }
        // ========================= Pthread改进部分 =========================

        // ========================= OpenMP改进部分  =========================
        int value_count = pt.max_indices[pt.content.size() - 1];

        if (value_count < PARALLEL_THREADSHOLD) 
        {
            // ================= 串行处理 =================
            for (int i = 0; i < value_count; ++i) 
            {
                string temp = guess + a->ordered_values[i];
                guesses.push_back(temp);
                total_guesses++;
            }
        } 
        else 
        {
            // ================= 并行处理 =================
            omp_set_num_threads(4);
            int thread_num = omp_get_max_threads();
            vector<vector<string>> thread_guesses(thread_num);
            vector<int> thread_counts(thread_num, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                vector<string>& local_vec = thread_guesses[tid];
                int local_count = 0;

                #pragma omp for schedule(static)
                for (int i = 0; i < value_count; i++) 
                {
                    string temp = guess + a->ordered_values[i];
                    local_vec.push_back(temp);
                    local_count++;
                }

                thread_counts[tid] = local_count;
            }

            // 共享资源写入
            int total = 0;
            for (int cnt : thread_counts) 
            {
                total += cnt;
            }
            total_guesses += total;

            guesses.reserve(guesses.size() + total);
            for (auto& vec : thread_guesses) 
            {
                guesses.insert(guesses.end(), vec.begin(), vec.end());
            }
        }
        // ========================= OpenMP改进部分  =========================
    }
    last_generated_count = guesses.size() - total_guesses_before;
}