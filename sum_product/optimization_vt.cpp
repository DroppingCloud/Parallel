#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>

using namespace std;

const int N = 18800;  // 矩阵大小上限
const int TIME_LIMIT = 500;   // 执行上限

double a[N], sum;
double sum1, sum2;

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

// 递归优化
void recursion(int n)
{
    for (int m = n; m > 1; m /= 2)
        for (int i = 0; i < m / 2; i++)
            a[i] = a[i * 2] + a[i * 2 + 1];
}
 
 // 多路链式优化
 __attribute__((noinline))void multi_link(int n)
{
    sum1 = 0; sum2 = 0; sum = 0;
    for (int i = 0;i < n; i += 2) 
    {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;
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
    for(int i = 0;i < TIME_LIMIT;i++)
    {
        // recursion(n);

        init(n);
        multi_link(n);

        counter++;
    }
    cout << sum <<endl;

    long long end = get_time_ms();

    double total_sec = (end - start) / 1000.0;

    // 输出表格数据
    cout << left << setw(15) << n
            << setw(10) << counter
            << setw(20) << fixed << setprecision(8) << total_sec
            << endl;

    return 0;
}




