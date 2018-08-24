% !TEX encoding = UTF-8 Unicode
% ------ Path to chapter files ----- %
\newcommand*{\chapters}{\basepath chapters/}

% ------ SubSubSubSection ----- %
\newcommand{\subsubsubsection}[1]{{\bf #1\par}}


% ------ Math Operations ----- %
\newcommand{\randomvariable}[1]{\mathfrak{#1}}
\newcommand{\set}[1]{\mathcal{#1}}
\newcommand{\hollow}[1]{\mathbb{#1}}
\newcommand{\mthfnc}[1]{ \mathit{#1} }
\newcommand{\simplex}[1]{ \Delta^{#1-1} = \left\lbrace x \in \hollow{R}_+^{#1} \; : \; \sum^{#1}_{i=1} x_i = 1 \right\rbrace }


% ------ Double Column Footnote ----- %
\twocolumnfootnotes

\footmarkwidth=1.8em

\makeatletter

\bidi@ensure@newcommand{\footfootmark}{%
  \ifdim\footmarkwidth < \z@
    \llap{\hb@xt@ -\footmarkwidth{%
            \hss\normalfont\footscript{\@thefnmark}}%
          \hspace*{-\footmarkwidth}}%
  \else
    \ifdim\footmarkwidth = \z@
      {\normalfont\footscript{\@thefnmark}}%
    \else
      \hb@xt@\footmarkwidth{\hss\normalfont\footscript{.\@thefnmark}}%
    \fi
  \fi}

\makeatother


% ------ Pseudo Code Environenment ----- %
\renewcommand{\algorithmicif}{\textbf{اگر}}
\renewcommand{\algorithmicwhile}{\textbf{تا زمانی که}}
\renewcommand{\algorithmicend}{\textbf{پایان}}
\renewcommand{\algorithmicfor}{\textbf{به ازای}}
\renewcommand{\algorithmicdo}{\textbf{انجام بده}}
\renewcommand{\algorithmicthen}{\textbf{آنگاه}}
\renewcommand{\algorithmicelse}{\textbf{وگرنه}}
\renewcommand{\algorithmicprint}{\textbf{چاپ کن}}



% ------ Theorem Environenment ----- %
\newtheorem{theorem}{قضیه}[section]
\newtheorem{definition}[theorem]{تعریف}
\newtheorem{contract}[theorem]{پیمان}
\newtheorem{proposition}[theorem]{گزاره}
\newtheorem{lemma}[theorem]{لم}
\newtheorem{remark}[theorem]{تذکر}
\newtheorem{example}[theorem]{نمونه}
\newtheorem{corollary}[theorem]{نتیجه}