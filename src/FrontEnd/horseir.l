%{
#include <iostream>

#include "HorseIR_Parser.hpp"
#include "Utils/Logger.h"

#define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;

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

bool         return tBOOL;
i8           return tI8;
i16          return tI16;
i32          return tI32;
i64          return tI64;
f32          return tF32;
f64          return tF64;
complex      return tCOMPLEX;
sym          return tSYMBOL;
str          return tSTRING;
list         return tLIST;
table        { if (_functionIdent) { yylval.string_val = new std::string(yytext); return tIDENTIFIER; } return tTABLE; }
d            return tDATE;

module       return tMODULE;
import       return tIMPORT;
def          return tDEF;
kernel       return tKERNEL;
check_cast   return tCHECKCAST;
return       return tRETURN;

`{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = new std::string(yytext);
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

