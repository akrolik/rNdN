%{
#include <iostream>

#include "HorseIR_Parser.hpp"
#include "Utils/Logger.h"

#define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;

#define TYPE_ACTION(TOKEN) { if (_functionIdent) { yylval.string_val = new std::string(yytext); return tIDENTIFIER; } return TOKEN; }
bool _functionIdent = false;

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

"("          { _functionIdent = false; return '('; }
")"          return ')';
"{"          return '{';
"}"          return '}';
"["          return '[';
"]"          return ']';
"<"          return '<';
">"          return '>';
"="          return '=';
":"          return ':';
"+"          return '+';
"-"          return '-';
"*"          return '*';
"?"          return '?';
"@"          { _functionIdent = true; return '@'; }
";"          return ';';
"."          return '.';
","          return ',';

bool         TYPE_ACTION(tBOOL);
i8           TYPE_ACTION(tI8);
i16          TYPE_ACTION(tI16);
i32          TYPE_ACTION(tI32);
i64          TYPE_ACTION(tI64);
f32          TYPE_ACTION(tF32);
f64          TYPE_ACTION(tF64);
clex         TYPE_ACTION(tCOMPLEX);
sym          TYPE_ACTION(tSYMBOL);
str          TYPE_ACTION(tSTRING);
month        TYPE_ACTION(tMONTH);
date         TYPE_ACTION(tDATE);
dt           TYPE_ACTION(tDATETIME);
minute       TYPE_ACTION(tMINUTE);
second       TYPE_ACTION(tSECOND);
time         TYPE_ACTION(tTIME);
func         TYPE_ACTION(tFUNCTION);
list         TYPE_ACTION(tLIST);
dict         TYPE_ACTION(tDICTIONARY);
enum         TYPE_ACTION(tENUM);
table        TYPE_ACTION(tTABLE);
ktable       TYPE_ACTION(tKTABLE);

module       return tMODULE;
import       return tIMPORT;
def          return tDEF;
kernel       return tKERNEL;
check_cast   return tCHECKCAST;
return       return tRETURN;

`{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = new std::string(yytext + 1);
	return tSYMBOLVAL;
}

0|({NONZERODIGIT}{DIGIT}*) {
	yylval.int_val = atol(yytext);
	return tINTVAL;
}

(0\.{DIGIT}*)|({NONZERODIGIT}{DIGIT}*\.{DIGIT}*) {
	yylval.float_val = atof(yytext);
	return tFLOATVAL;
}

{DIGIT}{DIGIT}{DIGIT}{DIGIT}\.{DIGIT}{DIGIT}\.{DIGIT}{DIGIT} {
	int yy, mm, dd;
	sscanf(yytext, "%d.%d.%d", &yy, &mm, &dd);
	yylval.int_val = yy*10000 + mm*100 + dd;
	return tDATEVAL;
}

\"[^"]*\" {
	char *val = strdup(yytext + 1);
	val[strlen(val) - 1] = 0;
	yylval.string_val = new std::string(val);
	return tSTRINGVAL;
}

{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = new std::string(yytext);
	return tIDENTIFIER;
}

. {
	Utils::Logger::LogError("(line " + std::to_string(yylineno) + ") Unexpected character " + yytext);
}

%%

