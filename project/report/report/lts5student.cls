%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Signal Processing Laboratory (LTS5) - EPFL      %
% LaTeX student report template                   %
% Authors:                                        %
%   D. Perdios – dimitris.perdios@epfl.ch         %
%   A. Besson – adrien.besson@epfl.ch             %
% v0.1 - 22.12.16                                 %
% Typeset configuration: pdfLaTeX + Biber         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lts5student}[2016/12/22 LTS5 student project report LaTeX class]

%\DeclareOption{master}{\@mastertrue}
%TODO: New options master thesis or semester project
%TODO: New options eprint (default), print (a4paper, twoside/oneside, openright/openany)
%TODO: New options draft or final
%TODO: newif ->	http://tex.stackexchange.com/questions/27803/when-to-use-in-an-if-statement
%				http://tex.stackexchange.com/questions/68356/how-to-create-conditionals-in-a-document-class-for-latex

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions*
\LoadClass{report}

%% USEFUL REMARKS
% Guide to class: http://www.latex-project.org/guides/clsguide.pdf
% Difference between \let and \def resp. \(re)newcommand and \(re)newcommand* (http://tex.stackexchange.com/questions/1050/whats-the-difference-between-newcommand-and-newcommand)
%	http://tex.stackexchange.com/questions/655/what-is-the-difference-between-def-and-newcommand
%\def\examplea#1{% 				#1 cannot contain \par
%}
%\long\def\exampleb#1{% 			#1 can contain \par
%}
%\newcommand{\examplea}[1]{% 	#1 can contain \par
%}
%\newcommand*{\exampleb}[1]{% 	#1 cannot contain \par
%}
% \quad LaTeX macro is defined as \def\quad{\hskip1em\relax}
%TODO: change the \def and \let with \(re)newcommand* and \(re)newcommand for more robustness


%% PACKAGES
%TODO: continue package loading
%TODO: remove unnecessary packages of .cls (to make the main.tex compilable with another class)

% Fonts
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern} % latin modern font
%\RequirePackage[scaled=.92]{helvet} % sans serif typo
\RequirePackage[scaled=.92]{sourcesanspro}
\RequirePackage{textcomp} % pro­vide many text sym­bols (such as baht, bul­let, copy­right, mu­si­cal­note, onequar­ter, sec­tion, and yen), in the TS1 en­cod­ing
\RequirePackage{csquotes} % pro­vides ad­vanced fa­cil­i­ties for in­line and dis­play quo­ta­tions (better to load when using biblatex)
\RequirePackage{xcolor}
%\RequirePackage[final,tracking=true]{microtype}

% Typeset
\RequirePackage{setspace}
	\onehalfspacing % 1.5 linespaceing
%TODO: 1.5 linespacing in the whole document without setspace package
\RequirePackage{enumitem}
	\setlist[enumerate]{label*=\arabic*.,topsep=5pt,partopsep=0pt,parsep=0pt,itemsep=2pt}
	\setlist[itemize]{topsep=5pt,partopsep=0pt,parsep=0pt,itemsep=2pt}
%TODO: implement these options without enumitem package

% Math
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
%TODO: check the difference with gather and equation
%\let\equation\gather
%\let\endequation\endgather
%\displaywidowpenalty=150
%\postdisplaypenalty=10

% Figures
\usepackage[font=small, labelfont=bf, format=plain, labelsep=space, figurename=Figure, tablename=Table, skip=5pt]{caption}
\usepackage[labelfont=rm, labelformat=parens, labelsep=space, skip=0pt]{subcaption}
%TODO: change the citation style from 4.3a to 4.3 (a)

% Graphics
\RequirePackage{graphicx}

% Hypertext
% 	Hyperref is usually better to load at the very end...

%% LAYOUT
%TODO: if oneside or twoside (\@twosideoffsetmargin) -> seems to work for the rest of the variables even specified for twoside...
%TODO: lock a4paper option
%TODO: check if all those \@temp lengths are mandatory (see mitthesis.cls)
% 1 inch = 72.26999 pts
% Global margin
\newlength{\@totmargin}\setlength{\@totmargin}{1in} % desired global margin
% Horizontal spacing
\if@twoside
	\newlength{\@twosideoffsetmargin}\setlength{\@twosideoffsetmargin}{0.2in}
\else
	\newlength{\@twosideoffsetmargin}\setlength{\@twosideoffsetmargin}{0in}
\fi
\newlength{\@htotmargin}\setlength{\@htotmargin}{\@totmargin} % desired total vertical margin
\newlength{\@hbasismargin}\setlength{\@hbasismargin}{1in} % 1in internally defined !!do not modify!!
\setlength{\hoffset}{0pt}
\setlength{\marginparsep}{11pt} % default value (11pt)
\setlength{\marginparwidth}{36pt} % default value (111pt)
%	Compute odd- and evensidemargin
\newlength{\@temphmargin}\setlength{\@temphmargin}{\@htotmargin}
\addtolength{\@temphmargin}{-\@hbasismargin}
\newlength{\@tempoddsidemargin}\setlength{\@tempoddsidemargin}{\@temphmargin}
\newlength{\@tempevensidemargin}\setlength{\@tempevensidemargin}{\@temphmargin}
\addtolength{\@tempoddsidemargin}{\@twosideoffsetmargin}
\addtolength{\@tempevensidemargin}{-\@twosideoffsetmargin}
\setlength{\oddsidemargin}{\@tempoddsidemargin}
\setlength{\evensidemargin}{\@tempevensidemargin}
% 	Compute textwidth
\newlength{\@temptextwidth}\setlength{\@temptextwidth}{\paperwidth}
\addtolength{\@temptextwidth}{-2\@htotmargin}
\setlength{\textwidth}{\@temptextwidth}
% Vertical spacing
\newlength{\@vtotmargin}\setlength{\@vtotmargin}{\@totmargin} % desired total vertical margin
\newlength{\@vbasismargin}\setlength{\@vbasismargin}{1in} % 1in internally defined !!do not modify!!
\setlength{\voffset}{0pt} % default value is 0pt
\setlength{\headheight}{36pt} % fixed for multiple lines in header
\setlength{\headsep}{12pt} % fixed according to result with geometry package 1in margin
\setlength{\footskip}{30pt} % fixed according to result with geometry package 1in margin
%	Compute textheight
\newlength{\@temptextheight}\setlength{\@temptextheight}{\paperheight}
\addtolength{\@temptextheight}{-2\@vtotmargin}
\setlength{\textheight}{\@temptextheight}
%	Compute topmargin: topmargin = vtotmargin - vbasismargin(1in) - headheight - headsep 
\newlength{\@temptopmargin}\setlength{\@temptopmargin}{\@vtotmargin}
\addtolength{\@temptopmargin}{-\@vbasismargin}
\addtolength{\@temptopmargin}{-\voffset}
\addtolength{\@temptopmargin}{-\headheight}
\addtolength{\@temptopmargin}{-\headsep}
\setlength{\topmargin}{\@temptopmargin}

%% HEADERS
%TODO: check when title chapter title on a header is longer than one line
%TODO: check mitthesis.cls, reports.cls and fancyhdr.sty to make internally the result given by headers_settings.tex
%TODO: alignment left and center (see: http://tex.stackexchange.com/questions/55472/how-to-make-text-aligned-left-center-right-in-the-same-line)
%TODO: myheadings, check the difference with headings
\if@twoside
	\renewcommand*{\ps@headings}{%
		\renewcommand*{\@oddfoot}{\footnotesize\hfil\thepage\hfil}%
		\renewcommand*{\@evenfoot}{\footnotesize\hfil\thepage\hfil}%
		\renewcommand*{\@evenhead}{\footnotesize\hfil\leftmark\hfil}% left marker -> chapter
		\renewcommand*{\@oddhead}{\footnotesize\hfil\rightmark\hfil}% right marker -> section
		\renewcommand{\@mkboth}{\markboth}
		\renewcommand*{\chaptermark}[1]{\markboth{\MakeUppercase{%
			\ifnum \c@secnumdepth >\m@ne% if not numbered (e.g. starred chapters)
				\thechapter\quad%
			\fi
    		##1}}{}}%
		\renewcommand*{\sectionmark}[1]{\markright{\MakeUppercase{%
			\ifnum \c@secnumdepth >\z@% if not numbered (e.g. starred sections)
				\thesection\quad%
			\fi
    		##1}}}%
	}%
\else
	\renewcommand*{\ps@headings}{%
		\renewcommand*{\@oddfoot}{\footnotesize\hfil\thepage\hfil}%
		\renewcommand*{\@oddhead}{\footnotesize\hfil\rightmark\hfil}% only right marker -> chapter
		\renewcommand{\@mkboth}{\markboth}
		\renewcommand*{\chaptermark}[1]{\markright{\MakeUppercase{%
			\ifnum \c@secnumdepth >\m@ne% if not numbered (e.g. starred chapters)
				\thechapter\quad%
			\fi
		##1}}}%
	}%
\fi

% Redefine empty and plain style (from latex.ltx)
\renewcommand*{\ps@empty}{%
	\renewcommand{\@mkboth}{\@gobbletwo}
	\renewcommand{\@oddhead}{\@empty}
	\renewcommand{\@oddfoot}{\@empty}
	\renewcommand{\@evenhead}{\@empty}
	\renewcommand{\@evenfoot}{\@empty}
}%
\renewcommand*{\ps@plain}{%
	\renewcommand{\@mkboth}{\@gobbletwo}
	\renewcommand{\@oddhead}{\@empty}
	\renewcommand*{\@oddfoot}{\footnotesize\hfil\thepage\hfil} % original: \reset@font = \normalfont
	\renewcommand{\@evenhead}{\@empty}
	\renewcommand{\@evenfoot}{\@oddfoot}
}%

% Redefine chapter pagestyle as empty and not plain to avoid headers
\renewcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
	\thispagestyle{plain}% original style: plain
	\global\@topnum\z@ % Prevents figures from going at top of page.
	\@afterindentfalse % Suppresses indent in first paragraph.
	\secdef\@chapter\@schapter}

% Redefine cleardoublepage pagestyle as empty and not plain to avoid headers
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
	\hbox{}%
	\thispagestyle{empty}%
	\newpage%
\if@twocolumn\hbox{}\newpage\fi\fi\fi}

%% SECTION TITLES
%TODO: find the exact reason why list of figures and list of tables doesn't have the same vertical spacing after the head than table of contents or chapters
%TODO: cleardoublepage definition, compare with mitthesis and http://tex.stackexchange.com/questions/161383/modify-style-of-cleardoublepage

% Redefine @makechapterhead and @makeschapterhead based on @makeschapterhead from report.cls
%TODO: space if chapter title spanning on multiple lines... check http://tex.stackexchange.com/questions/10326/how-to-set-the-chapter-style-in-report-class
%\def\@makechapterhead#1{%
%	\vspace*{50\p@}%
%	{\parindent \z@ \raggedright
%		\normalfont
%		\interlinepenalty\@M
%    	\Huge\bfseries
%    	\ifnum \c@secnumdepth >\m@ne
%    		\thechapter\quad
%    	\fi
%    	#1\par\nobreak
%		\vskip 40\p@
%	}%
%}
%\def\@makeschapterhead#1{%
%	\vspace*{50\p@}%
%	{\parindent \z@ \raggedright
%		\normalfont
%		\interlinepenalty\@M
%		\Huge \bfseries  #1\par\nobreak
%		\vskip 40\p@
%	}%
%}
% 	Patch using titlesec
\RequirePackage{titlesec}
\titleformat{\chapter}
	{\normalfont\Huge\bfseries}{\thechapter}{1em}{}
\titlespacing*{\chapter}{0pt}{50pt}{40pt}

%% TOC, LOT, LOF, LOA
%TODO: find why vertical spacing between LOT and LOF is not the same as TOC or a normal chapter

% Remove indentation of LOF and LOT exact definition (in latex.ltx)
\renewcommand*{\l@figure}{\@dottedtocline{1}{0em}{2.3em}}% Default: 1.5em/2.3em
\renewcommand{\l@table}{\l@figure}

% Algorithms
\RequirePackage{algorithm}
\renewcommand{\thealgorithm}{\thechapter.\arabic{algorithm}}
%	Patch using etoolbox, only for the indent -> http://tex.stackexchange.com/questions/80430/how-to-prevent-indent-on-listofalgorithms
%	Patch LOA to be typeset like LOT and LOF
\RequirePackage{etoolbox}
%	Patch vertical spacing, i.e. the spacing between LOA (chapter title) and first line + "lineskip" between entries in LOA (from different chapters)
%	The default \chapter inserts a 10pt vertical gap between chapter breaks in the LoF and LoT. Merely add a similar gap as part of the macro \@chapter to the LoA:
%	See: http://tex.stackexchange.com/questions/88562/the-line-spacing-in-the-list-of-algorithms
\patchcmd{\@chapter}% <cmd>
	{\chaptermark{#1}}% <search>
	{\chaptermark{#1}%
	\addtocontents{loa}{\protect\addvspace{10\p@}}}% replace
	{}{}% <success><failure>
%	Patch left indent of LOA by patching \listof command (used by float package which is loaded by algorithm)
%	See: http://tex.stackexchange.com/questions/58469/why-are-listof-and-listoffigures-styled-differently
\renewcommand*{\listof}[2]{%
	\@ifundefined{ext@#1}{\float@error{#1}}{%
		\expandafter\let\csname l@#1\endcsname \l@figure  % <- use layout of figure
		\float@listhead{#2}%
		\begingroup
			\setlength\parskip{0pt plus 1pt}%               % <- or drop this line completely
			\@starttoc{\@nameuse{ext@#1}}%
		\endgroup}}

%% MAKETITLE
%TODO: see mitthesis.cls class or http://tex.stackexchange.com/questions/125383/default-values-in-newcommand-renewing-maketitle
%TODO: multiple supervisors, maybe with \\ \hline like commands (mitthesis.cls -> \supervisor definition) + let empty
%TODO: laboratory entry + let empty
%TODO: laboratory logo?

%	Supervisor one is mandatory
%\newcommand*{\@supervisorlist{}}
%\def\supervisor#1#2{%
%	\def\@supervisorname{#1}%
%	\def\@supervisortitle{#2}%
%	\gdef\@supervisor{\ifx\@supervisortitle\@empty\else\@supervisortitle~\fi\@supervisorname}%
%}
%\newcommand{\supervisor}[2]{
%	\newcommand{\@supervisorname}{#1}
%	\newcommand{\@supervisortitle}{#2}
%	\renewcommand{\@supervisor}{
%		\ifdefempty{\@supervisortitle}{}{\@supervisortitle~}\@supervisorname
%	}
%}

%       Newcommand to define a second and third authors
% init as empty for use in maketitle with ifdefempty (etoolbox)
\newcommand{\@authortwo}{}
\newcommand{\authortwo}[1]{% <- newcommand with 1 optional argument and a total of 2 arguments
	\newcommand{\@authortwoname}{#1}% <- required "name" argument
	\renewcommand{\@authortwo}{
		\@authortwoname
	}
}
% init as empty for use in maketitle with ifdefempty (etoolbox)
\newcommand{\@authorthree}{}
\newcommand{\authorthree}[1]{% <- newcommand with 1 optional argument and a total of 2 arguments
	\newcommand{\@authorthreename}{#1}% <- required "name" argument
	\renewcommand{\@authorthree}{
		\@authorthreename
	}
}

%		Newcommand with 1 optional argument and a total of 2 arguments, the option is the #1 and the required is the #2
\newcommand{\supervisor}[2][]{% <- newcommand with 1 optional argument and a total of 2 arguments
	\newcommand{\@supervisorname}{#2}% <- required "name" argument
	\newcommand{\@supervisortitle}{#1}% <- option "title" argument
	\renewcommand{\@supervisor}{
		\ifdefempty{\@supervisortitle}{}{\@supervisortitle~}\@supervisorname
	}
}
\def\@supervisor{\@latex@warning@no@line{No \noexpand\supervisor given}}

%	Supervisor two is option
\newcommand{\@supervisortwo}{}% init as empty for use in maketitle with ifdefempty (etoolbox)
\newcommand{\supervisortwo}[2][]{% <- newcommand with 1 optional argument and a total of 2 arguments
	\newcommand{\@supervisortwoname}{#2}% <- required "name" argument
	\newcommand{\@supervisortwotitle}{#1}% <- option "title" argument
	\renewcommand{\@supervisortwo}{
		\ifdefempty{\@supervisortwotitle}{}{\@supervisortwotitle~}\@supervisortwoname
	}
}

%	Assistant one is mandatory
%\def\assistant#1#2{%
%	\def\@assistantname{#1}%
%	\def\@assistanttitle{#2}%
%	\gdef\@assistant{\ifx\@assistanttitle\@empty\else\@assistanttitle~\fi\@assistantname}%
%}
\newcommand{\assistant}[2][]{% <- newcommand with 1 optional argument and a total of 2 arguments
	\newcommand{\@assistantname}{#2}% <- required "name" argument
	\newcommand{\@assistanttitle}{#1}% <- option "title" argument
	\renewcommand{\@assistant}{
		\ifdefempty{\@assistanttitle}{}{\@assistanttitle~}\@assistantname
	}
}
\def\@assistant{\@latex@warning@no@line{No \noexpand\assistant given}}

%	Assistants two is optional
\newcommand{\@assistanttwo}{}% init as empty for use in maketitle with ifdefempty (etoolbox)
\newcommand{\assistanttwo}[2][]{% <- newcommand with 1 optional argument and a total of 2 arguments
	\newcommand{\@assistanttwoname}{#2}% <- required "name" argument
	\newcommand{\@assistanttwotitle}{#1}% <- option "title" argument
	\renewcommand{\@assistanttwo}{
		\ifdefempty{\@assistanttwotitle}{}{\@assistanttwotitle~}\@assistanttwoname
	}
}

%	Project type
\def\projecttype#1{\gdef\@projecttype{#1}}
\def\@projecttype{\@latex@warning@no@line{No \noexpand\projecttype given}}

% Newcommands
\newcommand{\eX}{\dimexpr\fontcharht\font`X\relax} % height of X for specific font when called
%TODO: exact size of 'X' not perfectly computed with helvet font whether for 'x' it is ok
\newcommand{\myrulethickness}{0.25\dimexpr\fontcharht\font`x\relax} % height of X for specific font when called
\newcommand\mytoprule{%
	\newlength{\mytoprulevpos}
	\setlength{\mytoprulevpos}{0ex}
%	\noindent\rule[\mytoprulevpos]{0.5\linewidth}{\myrulethickness}xMXSome\par\vspace{-\mytoprulevpos}% TEST command
	\noindent\rule[\mytoprulevpos]{\linewidth}{\myrulethickness}\par\vspace{-\mytoprulevpos}
}
\newcommand\mybottomrule{%
	\newlength{\mybottomrulevpos}
	\setlength{\mybottomrulevpos}{\eX - \myrulethickness}
%	\noindent\rule[\mybottomrulevpos]{0.5\linewidth}{\myrulethickness}xMXSome\par\vspace{-\mybottomrulevpos}% TEST command
	\noindent\rule[\mybottomrulevpos]{\linewidth}{\myrulethickness}\par\vspace{-\mybottomrulevpos}
}
%TODO: check that the newlengths and newcommands are only local and not global values (see booktabs.dtx)

% Maketitle command
\renewcommand\maketitle{%
\begin{titlepage}
\centering
\renewcommand{\baselinestretch}{1.2} % global for the whole maketitle (including table)
\includegraphics[width=0.25\textwidth]{figures/logos/epfl.pdf}
% non-breaking space
\vfill
{% <-- begin group (definitions are local)
	\LARGE\sffamily\bfseries
	\mytoprule
	\uppercase\expandafter{\@title}\par
	\mybottomrule
}% <-- end group (definitions are local)
{%
	\Large\vspace{\baselineskip}
	{\@projecttype} for the course\par\vspace{\baselineskip}
	MATH-\num{412} Statistical Machine Learning,\par
	Ecole Polytechnique Federale de Lausanne,\par
	Fall \num{2017}\par\vspace{\baselineskip}
}%
{%
\renewcommand{\arraystretch}{1} % secures \arraystretch at 1 if other definitions in package.
\Large
\centering
\begin{tabular}{ll}
	\textsf{\textbf{Authors:}} & \@author, \@authortwo, \@authorthree \\
	
%	\textsf{\textbf{Supervisor:}} & \@supervisor \\
	\textsf{\textbf{Professor:}} & \@supervisor \\
\end{tabular}\par
}%
\vspace{\stretch{2}} % 1/3 - 2/3 stretch parameter
{\Large \@date\par}
\end{titlepage}}