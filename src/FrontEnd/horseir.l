%{
#include <stdio.h>
#include <string.h>

#include "HorseIR_Parser.hpp"

#define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;
%}

NONZERODIGIT [1-9]
DIGIT [0-9]
LETTER [A-Za-z_]

%option yylineno
%option noinput
%option nounput
%option noyywrap

%%

[ \t\r\n]+   /* ignore */

"("          return '(';
")"          return ')';
"{"          return '{';
"}"          return '}';
"<"          return '<';
">"          return '>';
"="          return '=';
":"          return ':';
"?"          return '?';
";"          return ';';
"."          return '.';
","          return ',';

i8           return tI8;
i64          return tI64;
string       return tSTRING;
list         return tLIST;
table        return tTABLE;

module       return tMODULE;
import       return tIMPORT;
def          return tDEF;
check_cast   return tCHECKCAST;
return       return tRETURN;

`{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = strdup(yytext);
	return tSYMBOL;
}

@{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = strdup(yytext);
	return tFUNCTION;
}

0|({NONZERODIGIT}{DIGIT}*) {
	yylval.int_val = atol(yytext);
	return tINTVAL;
}

\"({LETTER}|{DIGIT})*\" {
	yylval.string_val = strdup(yytext);
	return tSTRINGVAL;
}

{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = strdup(yytext);
	return tIDENTIFIER;
}

. {
	fprintf(stderr, "[Error] (line %d) Unexpected character %s\n", yylineno, yytext);
	exit(1);
}

%%

