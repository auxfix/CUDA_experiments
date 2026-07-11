#include <stdio.h>

int main() {
    int array_size = 10; 

    int arr_one[] = {10, 23, 76, 3, 76, -3, 0, 34, 12, 9};
    int arr_two[] = {5, 12, 45, 8, 34, -1, 2, 17, 6, 4};

    int sum_array[array_size];
    for (int i = 0; i < array_size; i++) {
        sum_array[i] = arr_one[i] + arr_two[i];
    }

    printf("CPU way of adding two arrays:\n");
    printf("Array One: ");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", arr_one[i]);
    }
    printf("\nArray Two: ");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", arr_two[i]);
    }
    printf("\nSum Array: ");
    for (int i = 0; i < array_size; i++) {
        printf("%d ", sum_array[i]);
    }
    printf("\n");
    return 0;
}