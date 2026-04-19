#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>


//方針
//スライドのprefixの分解をもう一段階多くする
//bucket[i]=iが何個あるか
//offset[i]=iは配列のどこに配置されるか

int main() {

/*元のコード
   int n = 5000000;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
*/

  double start_time = omp_get_wtime();
  int n = 50;//n=50000000
  int range = 5;//range=10000で効果がみられた
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
#pragma omp parallel for
  for (int i=0; i<n; i++)
#pragma omp atomic
    bucket[key[i]]++;


  std::vector<int> offset(range,0);
  std::vector<int> keepoffset(range);

  for(int i=1; i<range; i++)
    offset[i] = bucket[i-1];

//一段階目
#pragma omp parallel
{
  for(int j=1; j<range; j<<=1){
#pragma omp for
    for(int i=0; i<range; i++)
      keepoffset[i] = offset[i];
#pragma omp for
    for(int i=j; i<range; i++)
      offset[i] += keepoffset[i-j];
  }
}

//二段階目
  std::vector<int> currenst_offset = offset;
#pragma omp parallel for
  for(int i=0; i<range; i++){
    int start = offset[i];
    int count = bucket[i];
    for (int j=0; j<count; j++)
      key[start+j] = i;
  }


  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  double end_time = omp_get_wtime();

  printf("Items: %d\n", n);
  printf("Threads: %d\n", omp_get_max_threads());
  printf("Time: %f seconds\n", end_time - start_time);

  return 0;

}


//original output = 3 1 2 0 3 0 1 2 4 1 2 2 0 4 3 1 0 1 2 1 1 3 2 4 2 0 2 3 2 0 4 2 2 3 4 2 3 1 1 2 4 3 1 4 4 2 3 4 0 0
//                  0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4