export function timestamp() {
  return new Date().getTime();
}

export async function sleep(time: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, time));
}
