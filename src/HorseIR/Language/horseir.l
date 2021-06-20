%{
#include <iostream>

#include "HorseIR_Parser.hpp"
#include "Utils/Logger.h"

#include "HorseIR/Tree/Tree.h"

#define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;

bool _functionIdent = false;

#define TYPE_RETURN(TOKEN) { if (_functionIdent) { yylval.string_val = new std::string(yytext); return tIDENTIFIER; } return TOKEN; }
#define RETURN(TOKEN) { _functionIdent = false; return TOKEN; }

char UnescapeChar(char c, bool string)
{
	switch (c)
	{
		case 'a':
			return '\a';
		case 'b':
			return '\b';
		case 'f':
			return '\f';
		case 'n': 
			return '\n';
		case 'r':
			return '\r';
		case 't':
			return '\t';
		case 'v':
			return '\v';
		case '\\':
			return '\\';
		case '\"':
			if (string) {
				return '\"';
			}
			break;
		case '\'':
			if (!string) {
				return '\'';
			}
			break;
	}
	Utils::Logger::LogError("(line " + std::to_string(yylineno) + ") Unexpected escape character '\\" + c + "'");
}

std::string *UnescapeString(char *text, int len)
{
	int j = 0;
	char *string_val = (char *)malloc(len - 1);
	for (size_t i = 1; i < len - 1; i++, j++)
	{
		if (text[i] == '\\')
		{
			i++;
			string_val[j] = UnescapeChar(text[i], true);
		}
		else
		{
			string_val[j] = text[i];
		}
	}
	string_val[j] = '\0';
	auto s = new std::string(string_val);
	free(string_val);
	return s;
}

%}

 /* Reference: Hanfeng Chen, Hongji Chen (https://github.com/Sable/HorsePower/) */

NONZERODIGIT [1-9]
DIGIT [0-9]
LETTER [a-zA-Z_]

INT 0|({NONZERODIGIT}{DIGIT}*)
FLOAT ({INT}\.{DIGIT}*)|(\.{DIGIT}+)
STRING \"([^"\n\r\\]|(\\.))*\" 

YEAR   {DIGIT}{DIGIT}{DIGIT}{DIGIT}
MONTH  {DIGIT}{DIGIT}
DAY    {DIGIT}{DIGIT}
HOUR   {DIGIT}{DIGIT}
MINUTE {DIGIT}{DIGIT}
SECOND {DIGIT}{DIGIT}
MILLISECOND {DIGIT}{DIGIT}{DIGIT}

%option yylineno
%option noinput
%option nounput
%option noyywrap

%x COMMENT

%%

 /* Ignore whitespace and line/block comments */

[ \t\r\n]+   /* ignore */
\/\/.*       /* ignore */

\/\*                        { BEGIN(COMMENT); }
<COMMENT>\*\/               { BEGIN(INITIAL); }
<COMMENT>.                  { }
<COMMENT>\n                 { }
<COMMENT><<EOF>>            { Utils::Logger::LogError("(line " + std::to_string(yylineno) +") Unterminated block comment"); }

 /* Syntactic structural elements */

"("          RETURN('(');
")"          RETURN(')');
"{"          RETURN('{');
"}"          RETURN('}');
"["          RETURN('[');
"]"          RETURN(']');
"<"          RETURN('<');
">"          RETURN('>');
"="          RETURN('=');
":"          RETURN(':');
"+"          RETURN('+');
"-"          RETURN('-');
"?"          RETURN('?');
";"          RETURN(';');
","          RETURN(',');
"@"          { _functionIdent = true; return '@'; }
"."          return '.';
"*"          RETURN('*');
"/"          RETURN('/');

 /* Type reserved keywords */

bool         TYPE_RETURN(tBOOL);
char         TYPE_RETURN(tCHAR);
i8           TYPE_RETURN(tI8);
i16          TYPE_RETURN(tI16);
i32          TYPE_RETURN(tI32);
i64          TYPE_RETURN(tI64);
f32          TYPE_RETURN(tF32);
f64          TYPE_RETURN(tF64);
complex      TYPE_RETURN(tCOMPLEX);
sym          TYPE_RETURN(tSYMBOL);
str          TYPE_RETURN(tSTRING);
month        TYPE_RETURN(tMONTH);
date         TYPE_RETURN(tDATE);
dt           TYPE_RETURN(tDATETIME);
minute       TYPE_RETURN(tMINUTE);
second       TYPE_RETURN(tSECOND);
time         TYPE_RETURN(tTIME);
func         TYPE_RETURN(tFUNCTION);
list         TYPE_RETURN(tLIST);
dict         TYPE_RETURN(tDICTIONARY);
enum         TYPE_RETURN(tENUM);
table        TYPE_RETURN(tTABLE);
ktable       TYPE_RETURN(tKTABLE);

 /* Structural keywords */

module       RETURN(tMODULE);
import       RETURN(tIMPORT);
global       RETURN(tGLOBAL);
def          RETURN(tDEF);
kernel       RETURN(tKERNEL);
check_cast   RETURN(tCHECKCAST);
if           RETURN(tIF);
else         RETURN(tELSE);
while        RETURN(tWHILE);
repeat       RETURN(tREPEAT);
var          RETURN(tVAR);
return       RETURN(tRETURN);
break        RETURN(tBREAK);
continue     RETURN(tCONTINUE);

 /* Character literals, with escape characters [abfnrtv\'] */
 
'[^'\n\r\\]' {
	yylval.char_val = yytext[1];
	RETURN(tCHARVAL);
}

'\\.' {
	yylval.char_val = UnescapeChar(yytext[2], false);
	RETURN(tCHARVAL);
}

 /* Integer literal, no leading zero */
 
[+-]?{INT} {
	yylval.int_val = strtol(yytext, NULL, 10);
	RETURN(tINTVAL);
}

 /* Float literal, no leading zero, numbers on at least one side of the decimal */

[+-]?{FLOAT} {
	yylval.float_val = strtod(yytext, NULL);
	RETURN(tFLOATVAL);
}

 /* Imaginary value literal, optional real part */

[+-]?({INT}|{FLOAT})([+-]({INT}|{FLOAT}))?i {
	char *imag_start;
	double real = strtod(yytext, &imag_start);
	double imag = strtod(imag_start, NULL);
	yylval.complex_val = new HorseIR::ComplexValue(real, imag);
	RETURN(tCOMPLEXVAL);
}

 /* String literal, including support for escape characters [abfnrtv\"] */

{STRING} {
	yylval.string_val = UnescapeString(yytext, yyleng);
	RETURN(tSTRINGVAL);
}

 /* Date literals */

{YEAR}-{MONTH} {
	int year, month;
	sscanf(yytext, "%d-%d", &year, &month);
	yylval.month_val = new HorseIR::MonthValue(year, month);
	RETURN(tMONTHVAL);
}

{YEAR}-{MONTH}-{DAY} {
	int year, month, day;
	sscanf(yytext, "%d-%d-%d", &year, &month, &day);
	yylval.date_val = new HorseIR::DateValue(year, month, day);
	RETURN(tDATEVAL);
}

{HOUR}:{MINUTE} {
	int hour, minute;
	sscanf(yytext, "%d:%d", &hour, &minute);
	yylval.minute_val = new HorseIR::MinuteValue(hour, minute);
	RETURN(tMINUTEVAL);
}

{HOUR}:{MINUTE}:{SECOND} {
	int hour, minute, second;
	sscanf(yytext, "%d:%d:%d", &hour, &minute, &second);
	yylval.second_val = new HorseIR::SecondValue(hour, minute, second);
	RETURN(tSECONDVAL);
}

{HOUR}:{MINUTE}:{SECOND}\.{MILLISECOND} {
	int hour, minute, second, millisecond;
	sscanf(yytext, "%d:%d:%d.%d", &hour, &minute, &second, &millisecond);
	yylval.time_val = new HorseIR::TimeValue(hour, minute, second, millisecond);
	RETURN(tTIMEVAL);
}

{YEAR}-{MONTH}-{DAY}T{HOUR}:{MINUTE}:{SECOND}.{MILLISECOND} {
	int year, month, day;
	int hour, minute, second, millisecond;
	sscanf(yytext, "%d-%d-%dT%d:%d:%d.%d",
		&year, &month, &day,
		&hour, &minute, &second, &millisecond
	);
	yylval.datetime_val = new HorseIR::DatetimeValue(
		new HorseIR::DateValue(year, month, day),
		new HorseIR::TimeValue(hour, minute, second, millisecond)
	);
	RETURN(tDATETIMEVAL);
}

 /* Symbol literal */

`{LETTER}({LETTER}|{DIGIT})* {
	yylval.symbol_val = new HorseIR::SymbolValue(std::string(yytext + 1));
	RETURN(tSYMBOLVAL);
}

`{STRING} {
	auto s = UnescapeString(yytext + 1, yyleng - 1);
	yylval.symbol_val = new HorseIR::SymbolValue(*s);
	delete s;
	RETURN(tSYMBOLVAL);
}

 /* Identifier literal */

{LETTER}({LETTER}|{DIGIT})* {
	yylval.string_val = new std::string(yytext);
	RETURN(tIDENTIFIER);
}

. {
	Utils::Logger::LogError("(line " + std::to_string(yylineno) + ") Unexpected character " + yytext);
}

%%

