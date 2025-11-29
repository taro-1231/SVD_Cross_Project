void func1 ( AVThreadMessageQueue ** var1 ) 
{ 
if ( * var1 ) { 
func2 ( & ( * var1 ) -> var2 ) ; 
func3 ( & ( * var1 ) -> var3 ) ; 
func4 ( & ( * var1 ) -> var4 ) ; 
func5 ( var1 ) ; 
} 
} 
