#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

int main() {
    const int N = 16;
    float x[N], y[N], m[N], fx[N], fy[N];

    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0.0f;
    }


    /*ここから並列化*/
    for (int i = 0; i < N; i++) {

        // スカラのxiを16個に展開している
        __m512 xi = _mm512_set1_ps(x[i]);
        __m512 yi = _mm512_set1_ps(y[i]);
        // 一旦fx1, fy1を0で初期化
        __m512 fxi = _mm512_setzero_ps();
        __m512 fyi = _mm512_setzero_ps();

        // jループを16個ずつ処理していく
        for (int j = 0; j < N; j += 16) {

            // 配列から16個の値を一気にロード
            __m512 xj = _mm512_loadu_ps(&x[j]);
            __m512 yj = _mm512_loadu_ps(&y[j]);
            __m512 mj = _mm512_loadu_ps(&m[j]);


            // 計算の分割
            // rx = xi - xj, ry = yi - yj
            __m512 rx = _mm512_sub_ps(xi, xj);
            __m512 ry = _mm512_sub_ps(yi, yj);

            // r^2 = rx^2 + ry^2
            __m512 r2 = _mm512_fmadd_ps(rx, rx,
                           _mm512_mul_ps(ry, ry));

            // 1/r ≈ rsqrt
            __m512 inv_r = _mm512_rsqrt14_ps(r2);

            // 1/r^3 = inv_r^3
            __m512 inv_r3 = _mm512_mul_ps(inv_r,
                               _mm512_mul_ps(inv_r, inv_r));

            // 係数 m[j] * inv_r^3
            __m512 coef = _mm512_mul_ps(mj, inv_r3);

            // maskの作成(自分自身との相互作用を消す)
            __mmask16 mask = 0xFFFF;

            // i が [j, j+15] に含まれる場合だけ除外
            if (i >= j && i < j + 16) {
                mask &= ~(1 << (i - j));
            }

            // fx, fyの更新
            fxi = _mm512_mask_sub_ps(fxi, mask, fxi,
                    _mm512_mul_ps(rx, coef));
            fyi = _mm512_mask_sub_ps(fyi, mask, fyi,
                    _mm512_mul_ps(ry, coef));
        }

        // 最後の足すところ
        fx[i] = _mm512_reduce_add_ps(fxi);
        fy[i] = _mm512_reduce_add_ps(fyi);

        printf("%d %g %g\n", i, fx[i], fy[i]);
    }

    return 0;
}

/*
original output:
7 0.116756 -12.4161
8 -338.62 -57.6118
9 95.0369 113.957
10 -54.5483 16.0611
11 -8.91322 178.074
12 45.9304 17.5655
13 -5.91301 -10.5782
14 -79.2425 -72.9875
15 539.395 -22.1185
*/