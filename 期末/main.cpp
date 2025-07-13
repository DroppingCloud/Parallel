#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <deque>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

deque<std::vector<std::string>> hash_queue;
mutex hash_mutex;
condition_variable hash_cv;
bool producer_done = false;  // 控制哈希线程安全退出


// =================== 哈希线程函数 ===================
void hash_worker(double& time_hash, std::atomic<int>& hashed_cnt) {
    while (true) {
        std::vector<std::string> batch;
        {
            std::unique_lock<std::mutex> lock(hash_mutex);
            hash_cv.wait(lock, [] { return !hash_queue.empty() || producer_done; });
            if (hash_queue.empty() && producer_done) break;
            batch = std::move(hash_queue.front());
            hash_queue.pop_front();
        }
        // 计时
        auto start_hash = chrono::system_clock::now();
        bit32 state[4];
        // 可以并行哈希
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < batch.size(); ++i) {
            MD5Hash(batch[i], state); // SIMD可替换这里
            ++hashed_cnt;
        }
        auto end_hash = chrono::system_clock::now();
        auto duration = duration_cast<microseconds>(end_hash - start_hash);
        time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
    }
}

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    std::atomic<int> hashed_cnt{0}; // 可用于调试输出已哈希数量
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    // =================== 启动哈希线程 ===================
    std::thread hash_thread(hash_worker, std::ref(time_hash), std::ref(hashed_cnt));

    int curr_num = 0;
    int history = 0;
    const int PRINT_STEP = 100000;   // 每10w输出一次
    const int FLUSH_STEP = 1000000;  // 每100w批量哈希
    const int GUESS_LIMIT = 10000000;
    auto start = system_clock::now();

    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        // ========================= 查看猜测分布 =========================
        cout << "[PT DEBUG] ";
        cout << " Guesses generated: " << q.last_generated_count << endl;
        // ========================= 查看猜测分布 =========================

        cout << "[DEBUG] guesses = " << q.total_guesses << ", curr_num = " << curr_num << endl;

        // 每10w生成输出
        if (q.total_guesses - curr_num >= PRINT_STEP)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 判断是否超过猜测上限
            if (history + q.total_guesses > GUESS_LIMIT)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                break;
            }
        }

        // 每累计100w，异步推给哈希线程
        if (curr_num > FLUSH_STEP)
        {
            // === 关键变动：将一批待哈希口令送入队列，异步哈希 ===
            {
                std::lock_guard<std::mutex> lock(hash_mutex);
                hash_queue.push_back(std::move(q.guesses));
            }
            hash_cv.notify_one();

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    // while循环后，判断是否还有剩下未哈希的口令
    if (!q.guesses.empty()) {
        std::lock_guard<std::mutex> lock(hash_mutex);
        hash_queue.push_back(std::move(q.guesses));
        hash_cv.notify_one();
        history += curr_num; // 别忘了统计
        q.guesses.clear();
        curr_num = 0;
    }

    // 主循环结束后，通知哈希线程退出
    {
        std::lock_guard<std::mutex> lock(hash_mutex);
        producer_done = true;
    }
    hash_cv.notify_one();
    hash_thread.join();

    return 0;
}