#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>

using namespace std;

const int N = 2000;  // 矩阵大小上限
const int TIME_LIMIT = 100;   // 计时阈值（毫秒）

double a[N][N],b[N],sum[N] ;

// 初始化数据
void init(int n) 
{
    for (int i = 0; i < n; i++)
        b[i] = i + 1;

    for(int i = 0;i<n;i++)
        for(int j = 0;j<n;j++)
            a[i][j] = i+j+1;
}

// 获取当前时间
long long get_time_ms() 
{
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000LL + tv.tv_usec / 1000;  // 转换为毫秒
}

// 行优先算法
void compute(int n) 
{
    for (int i = 0;i<n;i++)
        sum[i] = 0.0;

    for (int j = 0; j < n; j++) 
        for (int i = 0; i < n; i++) 
            sum[i] += a[j][i] * b[j];  // 逐行访问, 一次外部循环向内积中累加一个乘法结果
}


int main() 
{
    // int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    cout << "Row" << endl;
    cout << left << setw(15) << "Matrix Size"
         << setw(10) << "Runs"
         << setw(20) << "Total Time (s)"
         << setw(25) << "Avg Time per Run (s)" << endl;
    cout << string(70, '-') << endl;

    for (int i = 0; i < num_sizes; i++) 
    {
        int n = sizes[i];
        init(n);

        int counter = 0;
        long long start = get_time_ms();

        // 重复实验以提高精度
        do 
        {
            compute(n);
            counter++;
        } while (get_time_ms() - start < TIME_LIMIT);

        long long end = get_time_ms();

        float total_sec = (end - start) / 1000.0;
        float avg_sec = total_sec / counter;

        // 输出表格数据
        cout << left << setw(15) << n
             << setw(10) << counter
             << setw(20) << fixed << setprecision(4) << total_sec
             << setw(25) << fixed << setprecision(8) << avg_sec
             << endl;
    }

    return 0;
}




