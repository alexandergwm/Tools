/*****************************************************************************
 * Generate_Sweep_Signal.c
 *****************************************************************************/

#include "adi_initialize.h"
#include "Generate_Sweep_Signal.h"

#include <sys/platform.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define FS (48000.0)
#define BLOCK_SIZE (240)
#define SIG_LEN (6.0)
#define SILENCE_LEN (2.0)
#define F_START (1.0)
#define F_END (24000.0)
#define FADE_IN (0.08)
#define FADE_OUT (0.005)
#define PI (3.14159265358979323846264338327)

#pragma pack(4)
typedef struct
{
    int Frame_Total;
    int FrameIdx_A;
    int FrameIdx_B;
    int FrameIdx_C;
    long double K;
    long double L;
    long double Dt;
    long double Di;
    long double Do;
    int FadeInSamples;      // 新增：淡入样本数
    int FadeOutSamples;     // 新增：淡出样本数
} GEN_SWEEP_SET;
#pragma pack()

GEN_SWEEP_SET Sweep_Set;

void Gen_Sweep_Init(void)
{
    // 计算总帧数
    Sweep_Set.Frame_Total = (int)((SIG_LEN + SILENCE_LEN) * FS / BLOCK_SIZE);

    // 计算关键帧索引
    Sweep_Set.FrameIdx_A = (int)(FADE_IN * FS / BLOCK_SIZE);
    Sweep_Set.FrameIdx_B = (int)((SIG_LEN - FADE_OUT) * FS / BLOCK_SIZE);
    Sweep_Set.FrameIdx_C = (int)(SIG_LEN * FS / BLOCK_SIZE);

    // 计算淡入淡出样本数（关键修正！）
    Sweep_Set.FadeInSamples = (int)(FADE_IN * FS);
    Sweep_Set.FadeOutSamples = (int)(FADE_OUT * FS);

    // 计算扫频参数
    Sweep_Set.K = (SIG_LEN * F_START * 2 * PI) / log(F_END / F_START);
    Sweep_Set.L = log(F_END / F_START) / SIG_LEN;

    // 时间步长
    Sweep_Set.Dt = 1.0 / FS;

    // 淡入淡出参数 - 关键修正！
    // 使用实际的样本数量，而不是帧索引
    Sweep_Set.Di = (PI / 2.0) / Sweep_Set.FadeInSamples;
    Sweep_Set.Do = (PI / 2.0) / Sweep_Set.FadeOutSamples;
}

int Gen_Sweep_Fun(int Frame_Num, long double *Buff)
{
    if (Frame_Num >= Sweep_Set.FrameIdx_C)
    {
        // 静音段
        memset(Buff, 0, BLOCK_SIZE * sizeof(long double));
        return 0;
    }

    // 计算当前帧的起始样本索引
    int start_sample_index = Frame_Num * BLOCK_SIZE;
    double start_time = start_sample_index * Sweep_Set.Dt;

    if (Frame_Num >= Sweep_Set.FrameIdx_A && Frame_Num < Sweep_Set.FrameIdx_B)
    {
        // 主扫频段（无淡入淡出）
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            long double t = start_time + i * Sweep_Set.Dt;
            long double phase = Sweep_Set.K * (expd(t * Sweep_Set.L) - 1.0);
            Buff[i] = (long double)sin(phase);
        }
    }
    else if (Frame_Num < Sweep_Set.FrameIdx_A)
    {
        // 淡入段 - 修正索引计算
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            long double t = start_time + i * Sweep_Set.Dt;
            int sample_index = start_sample_index + i;
            long double fade_factor = sin(sample_index * Sweep_Set.Di);
            long double phase = Sweep_Set.K * (expd(t * Sweep_Set.L) - 1.0);
            Buff[i] = (long double)(sin(phase) * fade_factor);
        }
    }
    else if (Frame_Num >= Sweep_Set.FrameIdx_B && Frame_Num < Sweep_Set.FrameIdx_C)
    {
        // 淡出段 - 修正索引计算
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
        	long double t = start_time + i * Sweep_Set.Dt;
            int sample_index = start_sample_index + i - Sweep_Set.FrameIdx_B * BLOCK_SIZE;
            long double fade_factor = cos(sample_index * Sweep_Set.Do);
            long double phase = Sweep_Set.K * (expd(t * Sweep_Set.L) - 1.0);
            Buff[i] = (long double)(sin(phase) * fade_factor);
        }
    }
    else
    {
        return -1;
    }

    return 0;
}


void Gen_Sweep_Main(void)
{
    Gen_Sweep_Init();

    FILE *fp = NULL;
    const char *filename = "out_Sweep.dat";

    fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Error: Cannot open file %s for writing!\n", filename);
        return;
    }

    printf("[%s] Write Start\n", filename);

    long double *out_buffer = (long double*)malloc(BLOCK_SIZE * sizeof(long double));
    if (!out_buffer)
    {
        printf("Error: Memory allocation failed!\n");
        fclose(fp);
        return;
    }

    int state = 0;
    int frames_written = 0;

    for (int i = 0; i < Sweep_Set.Frame_Total; i++)
    {
        state = Gen_Sweep_Fun(i, out_buffer);
        if (state == -1)
        {
            printf("Warning: Gen_Sweep_Error in frame [%d]\n", i);
            continue;
        }

        size_t written = fwrite(out_buffer, sizeof(long double), BLOCK_SIZE, fp);
        if (written != BLOCK_SIZE)
        {
            printf("Error: Write failed at frame [%d]\n", i);
            break;
        }

        frames_written++;

        // 进度显示
        if (i % 100 == 0)
        {
            printf("Processing: %d/%d frames (%.1f%%)\n",
                   i, Sweep_Set.Frame_Total,
                   (i * 100.0) / Sweep_Set.Frame_Total);
        }
    }

    free(out_buffer);
    fclose(fp);

    printf("[%s] Write completed! (%d frames, %.2f seconds)\n",
           filename, frames_written,
           frames_written * BLOCK_SIZE / FS);
}

int main(int argc, char *argv[])
{
    /**
     * Initialize managed drivers and/or services that have been added to
     * the project.
     * @return zero on success
     */
    adi_initComponents();

    printf("Starting sweep signal generation...\n");
    printf("Parameters:\n");
    printf("  Sample Rate: %.0f Hz\n", FS);
    printf("  Block Size: %d samples\n", BLOCK_SIZE);
    printf("  Signal Length: %.1f seconds\n", SIG_LEN);
    printf("  Silence Length: %.1f seconds\n", SILENCE_LEN);
    printf("  Frequency Range: %.1f Hz to %.1f Hz\n", F_START, F_END);
    printf("  Fade In: %.3f seconds\n", FADE_IN);
    printf("  Fade Out: %.3f seconds\n", FADE_OUT);

    Gen_Sweep_Main();

    printf("Sweep generation completed successfully!\n");

    return 0;
}
