clear
clc
% 时频域分析-emd分解，vmd分解
% 先把原始信号进行emd,vmd等方法分解，得到原始信号的本征模函数IMF
% EMD，VMD的目的是将组成原始信号的各尺度分量不断从高频到低频进行提取，
% 则分解得到的特征模态函数顺序是按频率由高到低进行排列的，
% 即首先得到最高频的分量，然后是次高频的，最终得到一个频率接近为0的残余分量。
% emd适合非线性、非平稳信号的分析，也适合于线性、平稳信号的分析，
% 并且对于线性、平稳信号的分析也比其他的时频分析方法更好地反映了信号的物理意义。
clear
clc
clf
fontsize =12;           %字体大小


Fs = 100;               %采样频率，即1s采多少个点
t = (0:1/Fs:10-1/Fs)';  %1000个采样点
f1 = 49;                %频率1
f2 = 20;                %频率2
x1 = 10*sin(2*pi*f1*t); %x1信号
x2 = 5*sin(2*pi*f2*t);  %x2信号
x = x1 + x2;            %较复杂信号，x信号由x1与x2组成
%问题？能不能对x进行处理，从而得到x1信号与x2信号？？
%emd作用通过x找到x1与x2分量                                            

                        
figure(1)               %时域波形
plot(t,x)
xlabel('时间/s')
ylabel('x信号时域幅值')


L = length(x);          %傅里叶变换
y = fft(x);
y = y/L;
P2 = abs(fft(x)/L);
P1 = P2(1:L/2);
P1(2:end-1) = 2*P1(2:end-1);
fnew = (0:(L/2-1))*Fs/L;
figure(2)
plot(fnew,P1)           %频域波形
xlabel('频率/s')
ylabel('x信号频域幅值')



% [imf,residual1] =emd(x,'MaxNumIMF',2);         %emd分解
[imf,residual1] =vmd(x);                       %vmd分解
figure(3)
ylabel('幅值','FontSize',fontsize)
xlabel('时间/s','FontSize',fontsize)
subplot(411)
plot(t,x)
ylabel('x原信号','FontSize',fontsize)
subplot(412)
plot(t,imf(:,1))
ylabel('IMF1','FontSize',fontsize)
subplot(413)
plot(t,imf(:,2))
ylabel('IMF2','FontSize',fontsize)
subplot(414)
plot(t,residual1)
ylabel('残差','FontSize',fontsize)


figure(4)
subplot(211)
plot(t,x1)
ylabel('x1原信号幅值')
subplot(212)
plot(t,imf(:,1))
ylabel('IMF1幅值')


figure(5)
subplot(211)
plot(t,x2)
ylabel('x2原信号幅值')
subplot(212)
plot(t,imf(:,2))
ylabel('IMF2幅值')

