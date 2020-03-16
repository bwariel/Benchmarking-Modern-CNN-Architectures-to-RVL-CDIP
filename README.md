# Benchmarking-Modern-CNN-Architectures-to-RVL-CDIP

This is repo associated with above paper. Pleas also see my (somewhat dated Medium Article) [Benchmarking Modern CNN Architectures to RVL-CDIP](https://medium.com/@jdegange85/benchmarking-modern-cnn-architectures-to-rvl-cdip-9dd0b7ec2955)

\begin{table}[]

\centering
\caption{Model Performance Details }
\label{tab:my-table}
\begin{tabular}{|l|l|l|l|l|l|l|l|l|l|}
\hline
Model name & Batch Size &  Test Accuracy & Total Steps(K) & Image Size & Optimizer & LR & Cutout
    \\ 
\hline
EfficientNetB4 &64 & \textbf{0.9281} & 500 & 380 & SGD & 0.01 & Y 
    \\
\hline
InceptionResNetV2 &16 &  \textbf{0.9263} & 250 & 512 & SGD & 0.1 & N
    \\
\hline
EfficientNetB2 &64 &  0.9157 & 500 & 260 & SGD & 0.01 & Y \\ \hline
EfficientNetB0 &64 &  0.9053 & 500 & 224 & SGD & 0.01 & Y \\ \hline
EfficientNetB0 &32 &  0.9036 & 247.5 & 224 & SGD & 0.01 & Y \\ \hline
EfficientNetB0 &32 &  0.8983 & 145 & 224 & SGD & 0.01 & Y \\ \hline
EfficientNetB0 &64 &  0.8951 & 100 & 224 & SGD & 0.01 & Y \\ \hline
EfficientNetB0 &32 &  0.8921 & 192.5 & 224 & Adadelta & 1.00 & Y \\ \hline
\end{tabular}
\end{table}
