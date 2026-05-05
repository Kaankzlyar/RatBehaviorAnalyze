% 1. Hedef klasörü belirle (Senin bilgisayarındaki yola göre)
folderPath = 'D:\ProjectsD\ThesisWork\data\DLCfiltered\Kare\ASP';

% 2. Klasördeki tüm .mat dosyalarının listesini al
matFiles = dir(fullfile(folderPath, '*.mat'));

% 3. Her bir dosya için döngü başlat
for k = 1:length(matFiles)
    
    % Dosya adını ve tam yolunu al, ardından yükle
    baseFileName = matFiles(k).name;
    fullFileName = fullfile(folderPath, baseFileName);
    data = load(fullFileName);
    
    % Yeni bir figür penceresi oluştur
    figure('Name', baseFileName, 'NumberTitle', 'off');
    
    % Zaman (t) kaydedilmediği için X ekseni olarak veri uzunluğunu (frame) kullanıyoruz
    frames = 1:length(data.xc); 
    
    % --- Grafik 1: X ve Y Koordinatlarının Değişimi ---
    subplot(2,1,1);
    % sgolayfilt yerine standart movmean fonksiyonunu kullanıyoruz
    plot(frames, movmean(data.xc, 51), 'b', 'LineWidth', 1.5);
    hold on;
    plot(frames, movmean(data.yc, 51), 'r', 'LineWidth', 1.5);
    title(['X ve Y Koordinatları: ', baseFileName], 'Interpreter', 'none');
    xlabel('Frame (Kare)');
    ylabel('Piksel Konumu');
    legend('X Koordinatı', 'Y Koordinatı');
    hold off;
    
    % --- Grafik 2: 2 Boyutlu Yörünge (Trajectory) ---
    subplot(2,1,2);
    % sgolayfilt yerine standart movmean fonksiyonunu kullanıyoruz
    plot(movmean(data.xc, 51), movmean(data.yc, 51), 'k', 'LineWidth', 1.5);
    title('Hareket Yörüngesi (Trajectory)');
    xlabel('X Ekseni (Piksel)');
    ylabel('Y Ekseni (Piksel)');
    
    % Videolarda (0,0) noktası sol üst köşe olduğu için Y eksenini ters çevirmek mantıklı olur
    set(gca, 'YDir', 'reverse'); 
    axis equal; % Oranları bozmamak için
    
end