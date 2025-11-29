int func1 ( AVThreadMessageQueue * var1 , 
void * var2 , 
unsigned var3 ) 
{ 
int var4 ; 
func2 ( & var1 -> var5 ) ; 
var4 = func3 ( var1 , var2 , var3 ) ; 
func4 ( & var1 -> var5 ) ; 
return var4 ; 
return AVERROR ( ENOSYS ) ; 
} 
