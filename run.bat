@REM @echo off
@REM :: 定义算法列表，用空格隔开
@REM @REM  mappo iddpg isac masac maddpg

@REM set algorithms=ippo
@REM if not exist logs mkdir logs

@REM for %%a in (%algorithms%) do (
@REM     echo ==========================================
@REM     echo Currently running: %%a ... Logging to logs\%%a.log
@REM     echo ==========================================
    
@REM     :: > logs\%%a.log 2>&1 表示将标准输出和错误输出都写入该文件
@REM     python benchmarl/run.py algorithm=%%a task=fheragent/hard_maze_static experiment=fheragent/fheragenthard_experiment.yaml seed=42 > logs\%%a.log 2>&1
@REM )

@REM echo All algorithms have been executed.
@REM pause

@REM 42 0 

@echo off
setlocal enabledelayedexpansion
@REM mappo iddpg isac masac  maddpg
:: 1. 定义参数列表（用空格隔开）
set algorithms=ippo iddpg
set tasks= fheragent/hard_maze_static
set seeds=0 42 3407

if not exist logs mkdir logs

:: 2. 开始嵌套循环
for %%t in (%tasks%) do (

    :: 提取任务名称最后一部分
    for %%f in (%%t) do (
        set task_name=%%~nxf
    )

    for %%a in (%algorithms%) do (
        for %%s in (%seeds%) do (
            
            echo ==========================================
            echo Task: %%t
            echo Algorithm: %%a
            echo Seed: %%s
            echo Logging to logs\!task_name!_%%a_%%s.log
            echo ==========================================
            echo running ================

            python benchmarl/run.py ^
                algorithm=%%a ^
                task=%%t ^
                experiment=fheragent/fheragenthard_experiment.yaml ^
                seed=%%s ^
                > logs\!task_name!_%%a_%%s.log 2>&1
        )
    )
)

echo.
echo All combinations have been executed.
pause
