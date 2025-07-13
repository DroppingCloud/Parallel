#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 一般编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

// MPI 编译指令如下
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

// CUDA 编译指令如下
// nvcc main.cpp train.cpp guessing.cpp md5.cpp -o main
// nvcc main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// nvcc main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main(int argc, char** argv)
{
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);     // 获取当前进程的rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);     // 获取总进程数

    // 所有进程都需要训练模型数据来参与Generate函数的计算
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    
    // 主进程初始化变量和计时器
    int curr_num = 0;
    int history = 0;
    auto start = system_clock::now();
    
    if (world_rank == 0) {
        cout << "Here" << endl;
    }
    
    // ========================= 所有进程都参与主循环 =========================
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (world_rank == 0) {
            // ========================= 查看猜测分布 =========================
            // 输出当前 PT 及其生成的猜测数
            cout << "[PT DEBUG] ";
            cout << " Guesses generated: " << q.last_generated_count << endl;
            // ========================= 查看猜测分布 =========================

            cout << "[DEBUG] guesses = " << q.total_guesses << ", curr_num = " << curr_num << endl;

            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " <<history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                // 在此处更改实验生成的猜测上限
                int generate_n=10000000;
                if (history + q.total_guesses > 10000000)
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
            // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
            // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
            // ================================串行部分=================================
            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (string pw : q.guesses)
                {
                    // TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
                    MD5Hash(pw, state);

                    // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
                    // a<<pw<<"\t";
                    // for (int i1 = 0; i1 < 4; i1 += 1)
                    // {
                    //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
                    // }
                    // a << endl;
                }

                // 在这里对哈希所需的总时长进行计算
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

                // 记录已经生成的口令总数
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
            // ================================串行部分=================================
            // ================================SIMD部分=================================
            // if (curr_num > 1000000)
            // {
            //     auto start_hash = system_clock::now();

            //     // 批量处理
            //     int i = 0;
            //     while (i < q.guesses.size())
            //     {
            //         vector<string> tmp;
            //         for (int j = 0; j < 4 && i + j < q.guesses.size(); j++)
            //         {
            //             tmp.push_back(q.guesses[i + j]);
            //         }

            //         bit32 states_simd[4][4] = {0};
            //         MD5Hash_SIMD(tmp, states_simd);

            //         // for (int j = 0; j < tmp.size(); ++j)
            //         // {
            //         //     cout << tmp[j] << "\t";
            //         //     for (int k = 0; k < 4; ++k)
            //         //     {
            //         //         cout << std::setw(8) << std::setfill('0') << hex << states_simd[j][k];
            //         //     }
            //         //     cout << endl;
            //         // }

            //         i += 4;
            //     }

            //     // 在这里对哈希所需的总时长进行计算
            //     auto end_hash = system_clock::now();
            //     auto duration = duration_cast<microseconds>(end_hash - start_hash);
            //     time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            //     // 记录已经生成的口令总数
            //     history += curr_num;
            //     curr_num = 0;
            //     q.guesses.clear();
            // }
            // ================================SIMD部分=================================
        }
    }

    // 等待所有进程完成
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        cout << "All processes completed successfully." << endl;
    }

    MPI_Finalize();
    return 0;
}
