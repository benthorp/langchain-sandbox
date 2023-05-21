export function logWithTimestamp(message) {
  const timestamp = new Date().toISOString();
  
  if (typeof message === 'string') {
    console.log(`%c[${timestamp}] %c${message}`, 'color: gray', 'color: black');
  } else if (typeof message === 'object') {
    console.log(`%c[${timestamp}]`, 'color: gray', message);
  } else {
    console.log(`%c[${timestamp}] %c${message}`, 'color: gray', 'color: black');
  }
}