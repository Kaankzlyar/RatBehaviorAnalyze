clearvars;clc;clf;close all;
% Kutu Final Dosyaları İçin
[d_file,d_path] = uigetfile( ...
    {'*.mat',...
    'Veri Dosyası (*.avi)'},...
    'Deney Sonuçlarının Tutulduğu Veri Dosyasını Seçiniz.');
load([d_path,d_file]);
D = 3*60;
t = linspace(0,D,length(xc));

M = 1280; N = 720;
pnt1 = [320, 50];
pnt2 = [920, 50];
pnt3 = [920, 670];
pnt4 = [320, 670];
pnt5 = [420, 150];
pnt6 = [830, 150];
pnt7 = [830, 537];
pnt8 = [420, 537];

P = [pnt1;pnt2;pnt3;pnt4;pnt1];
P2 = [pnt5;pnt6;pnt7;pnt8;pnt5];
X = P(:,1)'; Y = P(:,2)';
X2 = P2(:,1)'; Y2 = P2(:,2)';

xc_filtered = sgolayfilt(xc,1,51);
yc_filtered = sgolayfilt(yc,1,51);

figure(1);
C = [linspace(0,1,6)',[linspace(0,1,3),linspace(1,0,3)]',linspace(1,0,6)'];
tt = [0 10 20 30 60 120 180];
plot(xc_filtered(1),yc_filtered(1),'bo','MarkerSize',4); hold on;
for ii = 1 : 6
    plot(xc_filtered((t>=tt(ii))&(t<tt(ii+1))),...
        yc_filtered((t>=tt(ii))&(t<tt(ii+1))),...
        'Color',C(ii,:),'LineStyle','-.','LineWidth',2);
end
plot(xc_filtered(end),yc_filtered(end),'ro','MarkerSize',4);
plot(X2,Y2,'LineWidth',3,'Color','k');
axis([min(X),max(X),min(Y),max(Y)]);
leg1 = 'Başlangıç Noktası';
leg2 = sprintf('%.0fs >= t < %.0fs',tt(1),tt(2));
leg3 = sprintf('%.0fs >= t < %.0fs',tt(2),tt(3));
leg4 = sprintf('%.0fs >= t < %.0fs',tt(3),tt(4));
leg5 = sprintf('%.0fs >= t < %.0fs',tt(4),tt(5));
leg6 = sprintf('%.0fs >= t < %.0fs',tt(5),tt(6));
leg7 = sprintf('%.0fs >= t < %.0fs',tt(6),tt(7));
leg8 = 'Bitiş Noktası';
legend(leg1,leg2,leg3,leg4,leg5,leg6,leg7,leg8)

mxc = ((xc_filtered-min(X2))/(max(X2)-min(X2)))*0.45;
myc = ((yc_filtered-min(Y2))/(max(Y2)-min(Y2)))*0.45;
mmc = sqrt(mxc.^2+myc.^2);
figure(2);
subplot(311), plot(t,mxc,'b-.','LineWidth',2);
axis([t(1),t(end),0,0.5]);
ylabel('X Yönündeki Hareket');
xlabel('Zaman (s)');
subplot(312), plot(t,myc,'b-.','LineWidth',2);
axis([t(1),t(end),0,0.5]);
ylabel('Y Yönündeki Hareket');
xlabel('Zaman (s)');
subplot(313), plot(t,sqrt(mxc.^2+myc.^2),'b-.','LineWidth',2);
axis([t(1),t(end),0,0.5]);
ylabel('Hareket');
xlabel('Zaman (s)');

vx = (diff(mxc)./diff(t));
vy = (diff(mxc)./diff(t));
vv = (diff(sqrt(mxc.^2+myc.^2))./diff(t));
figure(3);
subplot(311), plot(t(1:end-1),vx,'b-.','LineWidth',2);
axis([t(1),t(end-1),-0.5,0.5]);
ylabel('X Yönündeki Hız (m/s)');
xlabel('Zaman (s)');
subplot(312), plot(t(1:end-1),vy,'b-.','LineWidth',2);
axis([t(1),t(end-1),-0.5,0.5]);
ylabel('Y Yönündeki Hız (m/s)');
xlabel('Zaman (s)');
subplot(313), plot(t(1:end-1),vv,'b-.','LineWidth',2);
axis([t(1),t(end-1),-0.5,0.5]);
ylabel('Hız (m/s)');
xlabel('Zaman (s)');

fprintf('Maximum X Hızı = %.4f\n',max(vx))
fprintf('Maximum Y Hızı = %.4f\n',max(vy))
fprintf('Minimum X = %.4f\n',min(mxc))
fprintf('Maximum X = %.4f\n',max(mxc))
fprintf('Minimum Y = %.4f\n',min(myc))
fprintf('Maximum Y = %.4f\n',max(myc))
fprintf('X Yönünde Hareketin Enerjisi = %.4f\n',((mxc-mean(mxc))*(mxc-mean(mxc))')/numel(mxc))
fprintf('Y Yönünde Hareketin Enerjisi = %.4f\n',((myc-mean(myc))*(myc-mean(myc))')/numel(myc))
fprintf('Hareketin Enerjisi = %.4f\n',((mmc-mean(mmc))*(mmc-mean(mmc))')/numel(mmc))
Res = [max(vx),max(vy),...
    min(mxc),max(mxc),...
    min(myc),max(myc),...
    ((mxc-mean(mxc))*(mxc-mean(mxc))')/numel(mxc),...
    ((myc-mean(myc))*(myc-mean(myc))')/numel(myc),...
    ((mmc-mean(mmc))*(mmc-mean(mmc))')/numel(mmc)]';
disp(Res)
