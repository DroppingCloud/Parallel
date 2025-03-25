#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>

using namespace std;

const int N = 18800;  // 矩阵大小上限
const int TIME_LIMIT = 500;   // 计时阈值（毫秒）

double a[N], sum;

// 初始化数据
void init(int n) 
{
    for (int i = 0; i < n; i++)
        a[i] = i + 1;
}

// 获取当前时间
long long get_time_ms() 
{
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1000LL + tv.tv_usec / 1000;  // 转换为毫秒
}

// 平凡算法
void compute(int n) 
{
    sum = 0;
    for (int i = 0; i < n; i++) 
        sum += a[i];
}

int main() 
{
    cout << left << setw(15) << "Matrix Size"
         << setw(10) << "Runs"
         << setw(20) << "Total Time (s)"
         << setw(25) << "Avg Time per Run (s)" << endl;
    cout << string(70, '-') << endl;

    int n = 4096;
    init(n);

    int counter = 0;
    long long start = get_time_ms();

    // 重复实验以提高精度
    for (int i = 0; i < TIME_LIMIT; i++)
    {
        init(n); 
        compute(n);
        counter++;
    }


    long long end = get_time_ms();

    double total_sec = (end - start) / 1000.0;

    // 输出表格数据
    cout << left << setw(15) << n
            << setw(10) << counter
            << setw(20) << fixed << setprecision(8) << total_sec
            << endl;

    return 0;
}




