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
    double K;
    double L;
    double Dt;
    double Di;
    double Do;
} GEN_SWEEP_SET;
#pragma pack()

GEN_SWEEP_SET Sweep_Set;

void Gen_Sweep_Init(void)
{
    // 计算总帧数 - 与MATLAB保持一致
    Sweep_Set.Frame_Total = (int)((SIG_LEN + SILENCE_LEN) * FS / BLOCK_SIZE);

    // 计算关键帧索引 - 与MATLAB保持一致
    Sweep_Set.FrameIdx_A = (int)(FADE_IN * FS / BLOCK_SIZE);
    Sweep_Set.FrameIdx_B = (int)((SIG_LEN - FADE_OUT) * FS / BLOCK_SIZE);
    Sweep_Set.FrameIdx_C = (int)(SIG_LEN * FS / BLOCK_SIZE);

    // 计算扫频参数 - 与MATLAB保持一致
    Sweep_Set.K = (SIG_LEN * F_START * 2 * PI) / log(F_END / F_START);
    Sweep_Set.L = log(F_END / F_START) / SIG_LEN;

    // 时间步长
    Sweep_Set.Dt = 1.0 / FS;

    // 淡入淡出参数 - 修正计算方式
    Sweep_Set.Di = (PI / 2.0) / (Sweep_Set.FrameIdx_A * BLOCK_SIZE);
    Sweep_Set.Do = (PI / 2.0) / ((Sweep_Set.FrameIdx_C - Sweep_Set.FrameIdx_B) * BLOCK_SIZE);
}

int Gen_Sweep_Fun(int Frame_Num, float *Buff)
{
    if (Frame_Num >= Sweep_Set.FrameIdx_C)
    {
        // 静音段
        memset(Buff, 0, BLOCK_SIZE * sizeof(float));
        return 0;
    }

    // 计算当前帧的起始时间（与MATLAB保持一致）
    double start_time = Frame_Num * BLOCK_SIZE * Sweep_Set.Dt;

    if (Frame_Num >= Sweep_Set.FrameIdx_A && Frame_Num < Sweep_Set.FrameIdx_B)
    {
        // 主扫频段（无淡入淡出）
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            double t = start_time + i * Sweep_Set.Dt;
            double phase = Sweep_Set.K * (exp(t * Sweep_Set.L) - 1.0);
            Buff[i] = (float)sin(phase);
        }
    }
    else if (Frame_Num < Sweep_Set.FrameIdx_A)
    {
        // 淡入段 - 与MATLAB保持一致
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            double t = start_time + i * Sweep_Set.Dt;
            double sample_index = Frame_Num * BLOCK_SIZE + i;
            double fade_factor = sin(sample_index * Sweep_Set.Di);
            double phase = Sweep_Set.K * (exp(t * Sweep_Set.L) - 1.0);
            Buff[i] = (float)(sin(phase) * fade_factor);
        }
    }
    else if (Frame_Num >= Sweep_Set.FrameIdx_B && Frame_Num < Sweep_Set.FrameIdx_C)
    {
        // 淡出段 - 与MATLAB保持一致
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            double t = start_time + i * Sweep_Set.Dt;
            double sample_index = (Frame_Num - Sweep_Set.FrameIdx_B) * BLOCK_SIZE + i;
            double fade_factor = cos(sample_index * Sweep_Set.Do);
            double phase = Sweep_Set.K * (exp(t * Sweep_Set.L) - 1.0);
            Buff[i] = (float)(sin(phase) * fade_factor);
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

    float *out_buffer = (float*)malloc(BLOCK_SIZE * sizeof(float));
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

        size_t written = fwrite(out_buffer, sizeof(float), BLOCK_SIZE, fp);
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
