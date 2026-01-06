function linker_analyzer_main()
    clc;
    clear;
    
    % 设置路径 (默认为当前文件夹)
    path = 'E:\Code_Github\Generate_Sweep\CCES\Generate_Sweep_Signal\Debug'; 
    files = dir(fullfile(path, '*.map.xml'));
    
    if isempty(files)
        fprintf('未找到 .map.xml 文件。\n');
        return;
    end
    
    % 遍历处理每一个找到的 xml 文件
    for k = 1:length(files)
        fprintf('正在处理: %s...\n', files(k).name);
        parseLinkerMap(fullfile(path, files(k).name), 'linker_analysis.txt');
    end
end

function parseLinkerMap(xmlFile, outputFile)
    fid = fopen(outputFile, 'w');
    if fid == -1
        error('无法打开文件:%s', outputFile);
    end

    % 读取 XML
    try
        xmlDoc = xmlread(xmlFile);
    catch ME
        error('XML 读取错误: %s', ME.message);
    end

    docElement = xmlDoc.getDocumentElement();
    memoryNodes = docElement.getElementsByTagName('MEMORY');
    numMemories = memoryNodes.getLength();
    dualPrint(fid, '发现 %d 个内存段\n', numMemories);

    % 初始化 summary 数组
    % Cols: 1:ID, 2:Name, 3:Start, 4:End, 5:Width, 6:Used(Bytes), 7:Unused(Bytes), 8:TotalWords, 9:TotalKB
    memSummary = cell(numMemories, 9);
    
    for memIdx = 0:numMemories-1
        currentMem = memoryNodes.item(memIdx);
        memAttrs = getAttributesMap(currentMem); % 使用辅助函数获取属性Map
        
        memID = memAttrs('id');
        memName = memAttrs('name');
        startAddr = memAttrs('start_address');
        endAddr = memAttrs('end_address');
        widthHex = memAttrs('width');

        startDec = hex2dec(startAddr(3:end));
        endDec = hex2dec(endAddr(3:end));
        widthBits = hex2dec(widthHex(3:end));
        widthBytes = widthBits / 8;

        totalBytes = (endDec - startDec + 1);
        totalWords = totalBytes / widthBytes;
        totalKB = totalBytes / 1024;

        wordsUsed = hex2dec(memAttrs('words_used'));
        wordsUnused = hex2dec(memAttrs('words_unused'));

        calculatedWords = wordsUsed + wordsUnused;
        usagePercent = 0;
        if calculatedWords > 0
            usagePercent = (wordsUsed / calculatedWords) * 100;
        end
        
        % 存储数据 (统一转换为 Bytes 存储)
        memSummary{memIdx+1, 1} = memID;
        memSummary{memIdx+1, 2} = memName;
        memSummary{memIdx+1, 3} = startAddr;
        memSummary{memIdx+1, 4} = endAddr;
        memSummary{memIdx+1, 5} = widthBits;
        memSummary{memIdx+1, 6} = wordsUsed * widthBytes;   % Used Bytes
        memSummary{memIdx+1, 7} = wordsUnused * widthBytes; % Unused Bytes
        memSummary{memIdx+1, 8} = totalWords;
        memSummary{memIdx+1, 9} = totalKB;

        dualPrint(fid, '\n-------------------------------------------------\n');
        dualPrint(fid, '内存段 [ID=%s]: %s\n', memID, memName);
        dualPrint(fid, '起始地址: %s\n', startAddr);
        dualPrint(fid, '结束地址: %s\n', endAddr);
        dualPrint(fid, '位宽: %d bits (%d bytes/word)\n', widthBits, widthBytes);
        dualPrint(fid, '总大小: %.2f KB (%d words)\n', totalKB, totalWords);

        if wordsUsed > 0 || wordsUnused > 0
            dualPrint(fid, ' 已使用: %d words (%.2f KB) \n', wordsUsed, (wordsUsed * widthBytes) / 1024);
            dualPrint(fid, ' 未使用: %d words (%.2f KB) \n', wordsUnused, (wordsUnused * widthBytes) / 1024);
            dualPrint(fid, ' 使用率: %.2f%%\n', usagePercent);
        else
            dualPrint(fid, ' 使用信息: 未提供 (可能为保留段)\n');
        end

        % -------- OUTPUT_SECTION Analysis (大符号查找) --------
        outputSections = currentMem.getElementsByTagName('OUTPUT_SECTION');
        sectionCount = outputSections.getLength();
        largeSymbolCount = 0;

        for outIdx = 0 : sectionCount - 1
            currentOut = outputSections.item(outIdx);
            inputSections = currentOut.getElementsByTagName('INPUT_SECTION');
            
            for inIdx = 0:inputSections.getLength() - 1
                currentIn = inputSections.item(inIdx);
                
                % 查找 SYMBOL
                symbols = currentIn.getElementsByTagName('SYMBOL');
                for symIdx = 0 : symbols.getLength() - 1
                    currentSym = symbols.item(symIdx);
                    symAttrs = getAttributesMap(currentSym);
                    
                    if ~isKey(symAttrs, 'size') || ~isKey(symAttrs, 'name')
                        continue;
                    end
                    
                    sizeHex = symAttrs('size');
                    sizeValue = hex2dec(sizeHex(3:end)); % Bytes

                    if sizeValue < 1024
                        continue;
                    end

                    largeSymbolCount = largeSymbolCount + 1;

                    demangledNodes = currentSym.getElementsByTagName('DEMANGLED_NAME');
                    demangledName = '';
                    if demangledNodes.getLength() > 0
                        demangledNode = demangledNodes.item(0);
                        if demangledNode.hasChildNodes
                            demangledName = char(demangledNode.getTextContent());
                        end
                    end

                    if largeSymbolCount == 1
                        dualPrint(fid, ' -> 发现大符号 (>=1KB):\n');
                    end
                    
                    dualPrint(fid, '    |- %-30s (Addr: %s, Size: %.2f KB)\n', ...
                        symAttrs('name'), symAttrs('address'), sizeValue/1024);
                    
                    if ~isempty(demangledName)
                        % 简化显示，去除换行符
                        demangledName = regexprep(demangledName, '\s+', ' ');
                        dualPrint(fid, '    |  |- Demangled: %s\n', demangledName);
                    end
                end
            end
        end
        dualPrint(fid, ' 输出段数量: %d, 大符号数量: %d\n', sectionCount, largeSymbolCount);
    end

       % -------- 表格总结 --------
    dualPrint(fid, '\n\n================ 内存段总结表 ================\n');
    % 修改表格头：增加End Addr列，移除Width列
    dualPrint(fid, '| %-10s | %-12s | %-12s | %-12s | %-10s | %-12s | %-12s |\n', ...
        'ID', 'Name', 'Start Addr', 'End Addr', 'Used(KB)', 'Unused(KB)', 'Total(KB)');
    dualPrint(fid, '-------------------------------------------------------------------------------------------------------\n');

    for i = 1:size(memSummary, 1)
        usedKB = memSummary{i, 6} / 1024;
        unusedKB = memSummary{i, 7} / 1024;
        totalKB = memSummary{i, 9};
        
        % 修改：打印结束地址(第4列)替代原宽度(第5列)
        dualPrint(fid, '| %-10s | %-12s | %-12s | %-12s | %-10.2f | %-12.2f | %-12.2f |\n', ...
            memSummary{i,1}, ...
            memSummary{i,2}, ...
            memSummary{i,3}, ...  % Start Address
            memSummary{i,4}, ...  % End Address (新增)
            usedKB, ...
            unusedKB, ...
            totalKB);
    end
    dualPrint(fid, '-------------------------------------------------------------------------------------------------------\n');


    % -------- 分类统计 (针对ADSP-21569内存布局) --------
    % 初始化统计变量 (单位: KB)
    l1Stats     = struct('used', 0, 'unused', 0, 'total', 0, 'name', 'L1 Memory');
    l2Stats     = struct('used', 0, 'unused', 0, 'total', 0, 'name', 'L2 Memory');
    l2UCStats   = struct('used', 0, 'unused', 0, 'total', 0, 'name', 'L2 Uncached');
    l2BCStats   = struct('used', 0, 'unused', 0, 'total', 0, 'name', 'L2 Boot Code');
    otherStats  = struct('used', 0, 'unused', 0, 'total', 0, 'name', 'Other Memory');

    for i = 1:size(memSummary, 1)
        memName = lower(memSummary{i,2});
        usedBytes = memSummary{i,6};
        unusedBytes = memSummary{i,7};
        totalKB = memSummary{i,9};
        
        % 转换为 KB
        usedKB = usedBytes / 1024;
        unusedKB = unusedBytes / 1024;

        % 根据内存段名称分类
        if contains(memName, 'mem_block') || contains(memName, 'mem_iv_code')
            % L1内存块（包括中断向量表）
            l1Stats.used = l1Stats.used + usedKB;
            l1Stats.unused = l1Stats.unused + unusedKB;
            l1Stats.total = l1Stats.total + totalKB;
        elseif contains(memName, 'mem_l2_bw') && ~contains(memName, 'mem_l2uc') && ~contains(memName, 'mem_l2bc')
            % L2缓存内存
            l2Stats.used = l2Stats.used + usedKB;
            l2Stats.unused = l2Stats.unused + unusedKB;
            l2Stats.total = l2Stats.total + totalKB;
        elseif contains(memName, 'mem_l2uc')
            % L2非缓存内存
            l2UCStats.used = l2UCStats.used + usedKB;
            l2UCStats.unused = l2UCStats.unused + unusedKB;
            l2UCStats.total = l2UCStats.total + totalKB;
        elseif contains(memName, 'mem_l2bc')
            % L2引导代码内存
            l2BCStats.used = l2BCStats.used + usedKB;
            l2BCStats.unused = l2BCStats.unused + unusedKB;
            l2BCStats.total = l2BCStats.total + totalKB;
        else
            % 其他内存
            otherStats.used = otherStats.used + usedKB;
            otherStats.unused = otherStats.unused + unusedKB;
            otherStats.total = otherStats.total + totalKB;
        end
    end

    % -------- 打印分类统计结果 --------
    dualPrint(fid, '\n================ 内存分类使用率统计 ================\n');
    dualPrint(fid, 'ADSP-21569 内存使用分析:\n\n');
    
    % 打印每个分类
    printStats(fid, l1Stats);
    printStats(fid, l2Stats);
    printStats(fid, l2UCStats);
    printStats(fid, l2BCStats);
    printStats(fid, otherStats);
    
    % 总计
    totalUsed = l1Stats.used + l2Stats.used + l2UCStats.used + l2BCStats.used + otherStats.used;
    totalSize = l1Stats.total + l2Stats.total + l2UCStats.total + l2BCStats.total + otherStats.total;
    totalPercent = 0;
    if totalSize > 0
        totalPercent = (totalUsed / totalSize) * 100;
    end
    
    dualPrint(fid, '\n--------------------------------------------------\n');
    dualPrint(fid, '【总体统计】\n');
    dualPrint(fid, '  总容量: %.2f KB (%.2f MB)\n', totalSize, totalSize/1024);
    dualPrint(fid, '  总使用: %.2f KB\n', totalUsed);
    dualPrint(fid, '  总利用率: %.2f%%\n', totalPercent);
    
    % 添加L1 Block的详细统计
    dualPrint(fid, '\n================ L1 Block 详细分析 ================\n');
    for i = 1:size(memSummary, 1)
        memName = memSummary{i,2};
        if contains(lower(memName), 'mem_block') || contains(memName, 'mem_iv_code')
            usedKB = memSummary{i,6} / 1024;
            totalKB = memSummary{i,9};
            usagePercent = 0;
            if totalKB > 0
                usagePercent = (usedKB / totalKB) * 100;
            end
            
            if contains(memName, 'mem_iv_code')
                dualPrint(fid, '%-20s: %.2f/%.2f KB (%5.2f%%) [中断向量表]\n', ...
                    memName, usedKB, totalKB, usagePercent);
            elseif contains(memName, 'mem_block0')
                dualPrint(fid, '%-20s: %.2f/%.2f KB (%5.2f%%) [L1-Block 0 - 1.5 MBit]\n', ...
                    memName, usedKB, totalKB, usagePercent);
            elseif contains(memName, 'mem_block1')
                dualPrint(fid, '%-20s: %.2f/%.2f KB (%5.2f%%) [L1-Block 1 - 1.5 MBit, DM Cache]\n', ...
                    memName, usedKB, totalKB, usagePercent);
            elseif contains(memName, 'mem_block2')
                dualPrint(fid, '%-20s: %.2f/%.2f KB (%5.2f%%) [L1-Block 2 - 1 MBit, PM Cache]\n', ...
                    memName, usedKB, totalKB, usagePercent);
            elseif contains(memName, 'mem_block3')
                dualPrint(fid, '%-20s: %.2f/%.2f KB (%5.2f%%) [L1-Block 3 - 1 MBit, Instruction Cache]\n', ...
                    memName, usedKB, totalKB, usagePercent);
            else
                dualPrint(fid, '%-20s: %.2f/%.2f KB (%5.2f%%)\n', ...
                    memName, usedKB, totalKB, usagePercent);
            end
        end
    end

    fclose(fid);
    fprintf('分析完成，结果已保存至 %s\n', outputFile);
end

% -------- 辅助函数: 同时打印到屏幕和文件 --------
function dualPrint(fid, varargin)
    str = sprintf(varargin{:});
    fprintf('%s', str);      % 打印到 Command Window
    fprintf(fid, '%s', str); % 打印到文件
end

% -------- 辅助函数: 打印单行统计 --------
function printStats(fid, stats)
    usage = 0;
    if stats.total > 0
        usage = (stats.used / stats.total) * 100;
    end
    dualPrint(fid, '%-12s -> Used: %8.2f KB | Free: %8.2f KB | Total: %8.2f KB | Usage: %5.2f%%\n', ...
        stats.name, stats.used, stats.unused, stats.total, usage);
end

% -------- 辅助函数: 获取 XML 节点属性为 Map --------
function attrsMap = getAttributesMap(node)
    attrsMap = containers.Map;
    if node.hasAttributes
        attributes = node.getAttributes;
        numAttrs = attributes.getLength;
        for i = 0:numAttrs-1
            attr = attributes.item(i);
            attrsMap(char(attr.getName)) = char(attr.getValue);
        end
    end
end
